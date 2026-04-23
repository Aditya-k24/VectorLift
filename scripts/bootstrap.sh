#!/usr/bin/env bash
# =============================================================================
# VectorLift — Bootstrap Script
# =============================================================================
# Waits for all infrastructure services to be healthy, then:
#   1. Creates Elasticsearch index with VectorLift mappings
#   2. Creates Qdrant collection for dense vectors
#   3. Runs PostgreSQL schema migrations (SQLAlchemy create_all)
#   4. Creates Kafka topics
#   5. Prints a success summary
#
# Usage (from repo root):
#   bash scripts/bootstrap.sh
#   # or inside Docker:
#   docker compose exec api bash /app/scripts/bootstrap.sh
#
# Environment variables (all have sensible defaults for local Docker Compose):
#   ES_HOST            Elasticsearch host (default: localhost)
#   ES_PORT            Elasticsearch port (default: 9200)
#   QDRANT_HOST        Qdrant host (default: localhost)
#   QDRANT_PORT        Qdrant port (default: 6333)
#   POSTGRES_HOST      PostgreSQL host (default: localhost)
#   POSTGRES_PORT      PostgreSQL port (default: 5432)
#   POSTGRES_DB        PostgreSQL database (default: vectorlift)
#   POSTGRES_USER      PostgreSQL user (default: vectorlift)
#   POSTGRES_PASSWORD  PostgreSQL password (default: vectorlift_secret)
#   REDIS_HOST         Redis host (default: localhost)
#   REDIS_PORT         Redis port (default: 6379)
#   KAFKA_HOST         Kafka host (default: localhost)
#   KAFKA_PORT         Kafka port (default: 9092)
#   ES_INDEX           Elasticsearch index name (default: vectorlift_passages)
#   QDRANT_COLLECTION  Qdrant collection name (default: vectorlift_dense)
#   EMBEDDING_DIM      Vector dimension (default: 768)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ES_HOST="${ES_HOST:-${ELASTICSEARCH_HOST:-localhost}}"
ES_PORT="${ES_PORT:-${ELASTICSEARCH_PORT:-9200}}"
QDRANT_HOST="${QDRANT_HOST:-${QDRANT_HOST:-localhost}}"
QDRANT_PORT="${QDRANT_PORT:-${QDRANT_PORT:-6333}}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-vectorlift}"
POSTGRES_USER="${POSTGRES_USER:-vectorlift}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-vectorlift_secret}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
KAFKA_HOST="${KAFKA_HOST:-${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}}"
# Strip port from KAFKA_BOOTSTRAP_SERVERS if needed
KAFKA_ACTUAL_HOST="${KAFKA_HOST%%:*}"
KAFKA_ACTUAL_PORT="${KAFKA_HOST##*:}"

ES_INDEX="${ES_INDEX:-${ELASTICSEARCH_INDEX:-vectorlift_passages}}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-${QDRANT_COLLECTION:-vectorlift_dense}}"
EMBEDDING_DIM="${EMBEDDING_DIM:-768}"

# Kafka topics
KAFKA_TOPIC_QUERIES="${KAFKA_TOPIC_QUERIES:-vectorlift.queries}"
KAFKA_TOPIC_FEEDBACK="${KAFKA_TOPIC_FEEDBACK:-vectorlift.feedback}"
KAFKA_TOPIC_DOCUMENTS="${KAFKA_TOPIC_DOCUMENTS:-vectorlift.documents}"

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_info()    { echo -e "${BLUE}[INFO ]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK   ]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN ]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

divider() { echo -e "${BOLD}──────────────────────────────────────────────────${NC}"; }

# Wait until a service responds — loops with sleep 5, max iterations
wait_for_service() {
  local name="$1"
  local check_cmd="$2"
  local max_attempts="${3:-60}"
  local attempt=0

  log_info "Waiting for ${name} to be ready..."
  until eval "$check_cmd" > /dev/null 2>&1; do
    attempt=$(( attempt + 1 ))
    if [[ $attempt -ge $max_attempts ]]; then
      log_error "${name} did not become healthy after $((max_attempts * 5))s. Aborting."
      exit 1
    fi
    echo -n "."
    sleep 5
  done
  echo ""
  log_success "${name} is ready."
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
divider
echo -e "${BOLD}  VectorLift Bootstrap — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
divider

# ---------------------------------------------------------------------------
# 1. Wait for Elasticsearch
# ---------------------------------------------------------------------------
log_info "Checking Elasticsearch at http://${ES_HOST}:${ES_PORT} ..."
wait_for_service "Elasticsearch" \
  "curl -sf 'http://${ES_HOST}:${ES_PORT}/_cluster/health?wait_for_status=yellow&timeout=5s'"

# ---------------------------------------------------------------------------
# 2. Wait for Qdrant
# ---------------------------------------------------------------------------
log_info "Checking Qdrant at http://${QDRANT_HOST}:${QDRANT_PORT} ..."
wait_for_service "Qdrant" \
  "curl -sf 'http://${QDRANT_HOST}:${QDRANT_PORT}/readyz'"

# ---------------------------------------------------------------------------
# 3. Wait for PostgreSQL
# ---------------------------------------------------------------------------
log_info "Checking PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT} ..."
wait_for_service "PostgreSQL" \
  "curl -sf --max-time 4 'http://${POSTGRES_HOST}:${POSTGRES_PORT}/' || \
   pg_isready -h '${POSTGRES_HOST}' -p '${POSTGRES_PORT}' -U '${POSTGRES_USER}' -d '${POSTGRES_DB}' 2>/dev/null || \
   python3 -c \"
import socket, sys
s = socket.socket()
s.settimeout(3)
try:
    s.connect(('${POSTGRES_HOST}', ${POSTGRES_PORT}))
    sys.exit(0)
except:
    sys.exit(1)
finally:
    s.close()
\""

# ---------------------------------------------------------------------------
# 4. Wait for Redis
# ---------------------------------------------------------------------------
log_info "Checking Redis at ${REDIS_HOST}:${REDIS_PORT} ..."
wait_for_service "Redis" \
  "curl -sf --max-time 3 'http://${REDIS_HOST}:${REDIS_PORT}/' || \
   python3 -c \"
import socket, sys
s = socket.socket()
s.settimeout(3)
try:
    s.connect(('${REDIS_HOST}', ${REDIS_PORT}))
    sys.exit(0)
except:
    sys.exit(1)
finally:
    s.close()
\""

# ---------------------------------------------------------------------------
# 5. Wait for Kafka
# ---------------------------------------------------------------------------
log_info "Checking Kafka at ${KAFKA_ACTUAL_HOST}:${KAFKA_ACTUAL_PORT} ..."
wait_for_service "Kafka" \
  "python3 -c \"
import socket, sys
s = socket.socket()
s.settimeout(5)
try:
    s.connect(('${KAFKA_ACTUAL_HOST}', int('${KAFKA_ACTUAL_PORT}')))
    sys.exit(0)
except:
    sys.exit(1)
finally:
    s.close()
\""

divider
log_info "All services healthy. Starting setup..."
divider

# ---------------------------------------------------------------------------
# 6. Create Elasticsearch Index
# ---------------------------------------------------------------------------
log_info "Creating Elasticsearch index: ${ES_INDEX}"

ES_INDEX_RESPONSE=$(curl -sf -X PUT \
  "http://${ES_HOST}:${ES_PORT}/${ES_INDEX}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"settings\": {
      \"number_of_shards\": 1,
      \"number_of_replicas\": 0,
      \"analysis\": {
        \"analyzer\": {
          \"vectorlift_analyzer\": {
            \"type\": \"custom\",
            \"tokenizer\": \"standard\",
            \"filter\": [\"lowercase\", \"stop\", \"porter_stem\"]
          }
        }
      }
    },
    \"mappings\": {
      \"properties\": {
        \"doc_id\": {
          \"type\": \"keyword\",
          \"doc_values\": true
        },
        \"passage_id\": {
          \"type\": \"keyword\"
        },
        \"title\": {
          \"type\": \"text\",
          \"analyzer\": \"vectorlift_analyzer\",
          \"fields\": {
            \"keyword\": { \"type\": \"keyword\", \"ignore_above\": 512 }
          }
        },
        \"text\": {
          \"type\": \"text\",
          \"analyzer\": \"vectorlift_analyzer\"
        },
        \"url\": {
          \"type\": \"keyword\"
        },
        \"metadata\": {
          \"type\": \"object\",
          \"dynamic\": true
        },
        \"indexed_at\": {
          \"type\": \"date\",
          \"format\": \"strict_date_optional_time||epoch_millis\"
        }
      }
    }
  }" 2>&1) || {
  # Index may already exist
  if echo "$ES_INDEX_RESPONSE" | grep -q "resource_already_exists_exception"; then
    log_warn "Elasticsearch index '${ES_INDEX}' already exists — skipping."
  else
    log_error "Failed to create Elasticsearch index. Response: ${ES_INDEX_RESPONSE}"
    exit 1
  fi
}

log_success "Elasticsearch index '${ES_INDEX}' is ready."

# ---------------------------------------------------------------------------
# 7. Create Qdrant Collection
# ---------------------------------------------------------------------------
log_info "Creating Qdrant collection: ${QDRANT_COLLECTION}"

QDRANT_RESPONSE=$(curl -sf -X PUT \
  "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"vectors\": {
      \"size\": ${EMBEDDING_DIM},
      \"distance\": \"Cosine\"
    },
    \"optimizers_config\": {
      \"default_segment_number\": 2,
      \"memmap_threshold\": 20000
    },
    \"replication_factor\": 1,
    \"write_consistency_factor\": 1,
    \"on_disk_payload\": false
  }" 2>&1) || {
  if echo "$QDRANT_RESPONSE" | grep -qi "already exists"; then
    log_warn "Qdrant collection '${QDRANT_COLLECTION}' already exists — skipping."
  else
    log_error "Failed to create Qdrant collection. Response: ${QDRANT_RESPONSE}"
    exit 1
  fi
}

# Create payload indexes for efficient filtering
log_info "Creating Qdrant payload indexes..."
curl -sf -X PUT \
  "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}/index" \
  -H 'Content-Type: application/json' \
  -d '{"field_name": "doc_id", "field_schema": "keyword"}' > /dev/null 2>&1 || true

curl -sf -X PUT \
  "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}/index" \
  -H 'Content-Type: application/json' \
  -d '{"field_name": "passage_id", "field_schema": "keyword"}' > /dev/null 2>&1 || true

log_success "Qdrant collection '${QDRANT_COLLECTION}' is ready."

# ---------------------------------------------------------------------------
# 8. Run PostgreSQL Migrations (SQLAlchemy create_all via setup_db.py)
# ---------------------------------------------------------------------------
log_info "Running PostgreSQL schema migrations..."

if command -v python3 &> /dev/null; then
  POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  POSTGRES_HOST="${POSTGRES_HOST}" \
  POSTGRES_PORT="${POSTGRES_PORT}" \
  POSTGRES_DB="${POSTGRES_DB}" \
  POSTGRES_USER="${POSTGRES_USER}" \
    python3 -m scripts.setup_db 2>&1 || \
    python3 scripts/setup_db.py 2>&1 || {
    log_error "PostgreSQL migration failed."
    exit 1
  }
  log_success "PostgreSQL schema is up to date."
else
  log_warn "python3 not found — skipping DB migration. Run 'python scripts/setup_db.py' manually."
fi

# ---------------------------------------------------------------------------
# 9. Create Kafka Topics
# ---------------------------------------------------------------------------
log_info "Creating Kafka topics..."

create_kafka_topic() {
  local topic="$1"
  local partitions="${2:-3}"
  local replication="${3:-1}"

  python3 -c "
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import sys

client = KafkaAdminClient(
    bootstrap_servers='${KAFKA_ACTUAL_HOST}:${KAFKA_ACTUAL_PORT}',
    client_id='vectorlift-bootstrap',
    request_timeout_ms=10000,
)
topic = NewTopic(
    name='${topic}',
    num_partitions=${partitions},
    replication_factor=${replication},
)
try:
    client.create_topics([topic])
    print('Created topic: ${topic}')
except TopicAlreadyExistsError:
    print('Topic already exists: ${topic}')
except Exception as e:
    print(f'Error creating topic ${topic}: {e}', file=sys.stderr)
    sys.exit(1)
finally:
    client.close()
" 2>&1 || {
    log_warn "Could not create Kafka topic '${topic}' via Python. Trying kafka-topics CLI..."
    kafka-topics --bootstrap-server "${KAFKA_ACTUAL_HOST}:${KAFKA_ACTUAL_PORT}" \
      --create --if-not-exists \
      --topic "${topic}" \
      --partitions "${partitions}" \
      --replication-factor "${replication}" 2>/dev/null || \
    log_warn "Could not create topic '${topic}' — it may already exist or kafka-topics is not available."
  }
}

create_kafka_topic "${KAFKA_TOPIC_QUERIES}"   3 1
create_kafka_topic "${KAFKA_TOPIC_FEEDBACK}"  3 1
create_kafka_topic "${KAFKA_TOPIC_DOCUMENTS}" 3 1

log_success "Kafka topics are ready."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
divider
echo -e "${GREEN}${BOLD}"
echo "  VectorLift Bootstrap Complete!"
echo ""
echo "  Services:"
echo "    Elasticsearch : http://${ES_HOST}:${ES_PORT}"
echo "    Qdrant        : http://${QDRANT_HOST}:${QDRANT_PORT}"
echo "    PostgreSQL    : ${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
echo "    Redis         : ${REDIS_HOST}:${REDIS_PORT}"
echo "    Kafka         : ${KAFKA_ACTUAL_HOST}:${KAFKA_ACTUAL_PORT}"
echo ""
echo "  Indexes / Collections:"
echo "    ES Index      : ${ES_INDEX}"
echo "    Qdrant        : ${QDRANT_COLLECTION} (dim=${EMBEDDING_DIM})"
echo ""
echo "  Kafka Topics:"
echo "    ${KAFKA_TOPIC_QUERIES}"
echo "    ${KAFKA_TOPIC_FEEDBACK}"
echo "    ${KAFKA_TOPIC_DOCUMENTS}"
echo -e "${NC}"
divider
