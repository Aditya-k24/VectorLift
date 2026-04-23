# Deployment Guide

This document covers all deployment scenarios for VectorLift: local development with Docker Compose, production considerations, Kubernetes setup, scaling guidance, and backup strategies.

---

## Prerequisites

| Tool | Minimum Version | Notes |
|---|---|---|
| Docker | 24.0 | Needed for Docker Compose |
| Docker Compose | 2.20 | Built into Docker Desktop |
| kubectl | 1.27 | For Kubernetes deployments |
| Python | 3.11 | For local dev without Docker |

---

## Docker Compose (Local / Staging)

The `docker-compose.yml` at the repo root defines a complete 12-service stack. All services share the `vectorlift-network` bridge network and use named volumes for persistent data.

### Services

| Service | Image | Port(s) | Purpose |
|---|---|---|---|
| `api` | `Dockerfile.api` | 8000 | FastAPI search engine |
| `dashboard` | `Dockerfile.dashboard` | 8501 | Streamlit UI |
| `worker` | `Dockerfile.worker` | — | Kafka consumer / indexing |
| `elasticsearch` | `elasticsearch:8.11.0` | 9200, 9300 | BM25 search backend |
| `qdrant` | `qdrant/qdrant:v1.7.0` | 6333, 6334 | Dense vector backend |
| `postgres` | `postgres:16` | 5432 | Metadata / experiments |
| `redis` | `redis:7-alpine` | 6379 | Query result cache |
| `zookeeper` | `cp-zookeeper:7.5.0` | 2181 | Kafka coordination |
| `kafka` | `cp-kafka:7.5.0` | 9092 | Event streaming |
| `prometheus` | `prom/prometheus:2.48.0` | 9090 | Metrics collection |
| `grafana` | `grafana/grafana:10.2.0` | 3000 | Metrics visualization |

### Startup

```bash
# Copy environment template
cp .env.example .env

# Start all services (detached)
docker compose up -d

# Check health
docker compose ps

# Wait for all services to become healthy (~60–90 seconds first run)
watch docker compose ps

# Bootstrap indexes and collections
./scripts/bootstrap.sh
```

### Common Operations

```bash
# View all logs (follow)
make docker-logs

# View API logs only
make docker-logs-api

# Restart a specific service
docker compose restart api

# Rebuild and restart after code changes
docker compose up -d --build api dashboard

# Stop (preserves volumes)
make docker-down

# Stop + wipe all volumes (destructive — removes all data)
make docker-down-volumes
```

### Development Mode with Hot Reload

Start only infrastructure services via Docker, then run the API and dashboard locally with hot reload:

```bash
# Start only infrastructure
docker compose up -d elasticsearch qdrant redis postgres kafka zookeeper

# Run API with uvicorn hot reload
make run-api
# Uvicorn watches core/, retrieval/, apps/ for changes

# Run dashboard with Streamlit auto-reload
make run-dashboard
```

This workflow is much faster during active development: no Docker build step required for code changes.

---

## Environment Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and adjust.

### Critical Variables for Production

```bash
# Security
APP_ENV=prod
SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
DEBUG=false

# Change all default passwords
POSTGRES_PASSWORD=<strong-password>
# Grafana: set GF_SECURITY_ADMIN_PASSWORD in docker-compose.yml or as env var

# Models: point to fine-tuned checkpoints if available
BIENCODER_MODEL_PATH=/app/models/biencoder/final_model
CROSSENCODER_MODEL_PATH=/app/models/reranker/final_model

# Use GPU if available
BIENCODER_DEVICE=cuda
CROSSENCODER_DEVICE=cuda
```

### Service Endpoints (Internal Docker Network)

Inside the Docker Compose network, services communicate via container names:

```bash
ELASTICSEARCH_HOST=elasticsearch   # not localhost
QDRANT_HOST=qdrant
REDIS_HOST=redis
POSTGRES_HOST=postgres
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
```

These are pre-set in `docker-compose.yml` via the `environment` block and do not need to be changed in `.env` for Docker deployments.

---

## Production Considerations

### Resource Requirements

| Service | CPU | RAM | Storage |
|---|---|---|---|
| API (per instance) | 2–4 cores | 4–8 GB (model weights) | — |
| Elasticsearch | 4 cores | 8 GB (heap: 4 GB) | 50+ GB SSD |
| Qdrant | 2–4 cores | 4–8 GB | Depends on corpus size* |
| PostgreSQL | 1–2 cores | 2–4 GB | 20 GB |
| Kafka | 2 cores | 4 GB | 50 GB |
| Redis | 1 core | 1–4 GB | — |

*Qdrant memory for full MS MARCO (8.8M × 768d float32): ~27 GB without quantization; ~7 GB with INT8 quantization.

### Elasticsearch Tuning

For production indexing of the full MS MARCO corpus:

```bash
# In docker-compose.yml — increase heap for 8.8M passages
ES_JAVA_OPTS=-Xms4g -Xmx4g

# Disable swap
ulimits:
  memlock:
    soft: -1
    hard: -1
```

Shard configuration:
- Start with 1 primary shard, 1 replica for staging
- Scale to 3+ shards for the full corpus or multi-node production clusters

### Qdrant Production Configuration

Enable gRPC for lower latency in production:

```bash
QDRANT_PREFER_GRPC=true
QDRANT_GRPC_PORT=6334
```

Enable product quantization to reduce memory footprint:

```python
from qdrant_client.models import QuantizationConfig, ScalarQuantizationConfig

client.create_collection(
    collection_name="vectorlift_dense",
    vectors_config=VectorsConfig(size=768, distance=Distance.COSINE),
    quantization_config=QuantizationConfig(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    )
)
```

### Redis Caching

The default Redis config in `docker-compose.yml` uses `allkeys-lru` eviction with 256 MB max memory. For production:

```bash
# Increase if query volume is high
--maxmemory 2gb
--maxmemory-policy allkeys-lru
```

Cache TTL is controlled by `REDIS_TTL_SECONDS` (default: 3600). Reduce for frequently-updated corpora.

### API Scaling

The API is stateless (model weights loaded at startup, all external state in ES/Qdrant/Redis/PG). Horizontal scaling is straightforward:

```bash
# Run 4 Gunicorn workers (production mode)
make run-api-prod
# Uses: gunicorn -k uvicorn.workers.UvicornWorker --workers 4

# Or scale via Docker Compose (requires a load balancer in front)
docker compose up -d --scale api=3
```

For multi-instance deployments behind Nginx or a cloud load balancer, all instances share the same Redis cache and PostgreSQL — no sticky sessions required.

### TLS / HTTPS

For HTTPS in production, terminate TLS at a reverse proxy (Nginx, Traefik, or cloud load balancer) and forward plain HTTP to the API containers. Do not expose port 8000 directly.

Example Nginx config snippet:

```nginx
upstream vectorlift_api {
    server api:8000;
}

server {
    listen 443 ssl;
    server_name search.yourdomain.com;

    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;

    location / {
        proxy_pass http://vectorlift_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }
}
```

---

## Kubernetes

Kubernetes manifests live in `infra/k8s/`. They define a namespace, deployments, services, configmaps, and persistent volume claims for each component.

### Quickstart

```bash
# Create namespace
kubectl apply -f infra/k8s/namespace.yaml

# Apply all manifests
kubectl apply -f infra/k8s/

# Verify pods
kubectl get pods -n vectorlift

# Verify services
kubectl get svc -n vectorlift

# Port-forward API for local testing
kubectl port-forward svc/vectorlift-api 8000:8000 -n vectorlift
```

### Key Resources

**API Deployment** (`infra/k8s/api-deployment.yaml`):

```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

**Horizontal Pod Autoscaler** for the API:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vectorlift-api-hpa
  namespace: vectorlift
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vectorlift-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Storage Classes

Elasticsearch and Qdrant require fast persistent storage:

```yaml
# Use SSD-backed storage class
storageClassName: fast-ssd  # adjust for your cloud provider
accessModes:
  - ReadWriteOnce
resources:
  requests:
    storage: 100Gi
```

Cloud provider storage class names:
- GKE: `pd-ssd`
- EKS: `gp3`
- AKS: `managed-premium`

### Secrets Management

Do not store credentials in Kubernetes YAML files. Use Kubernetes Secrets:

```bash
kubectl create secret generic vectorlift-secrets \
  --from-literal=postgres-password=<strong-password> \
  --from-literal=secret-key=<hex-token> \
  -n vectorlift
```

For production, use an external secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault) with the External Secrets Operator.

---

## Scaling Guidance

### Ingestion Throughput

The Kafka consumer is the ingestion bottleneck. Scale by increasing consumer group parallelism:

```bash
# docker-compose.yml — scale worker service
docker compose up -d --scale worker=4
```

All worker instances in the same consumer group will partition the Kafka topic automatically.

### Search Throughput

The API is CPU/memory-bound (model inference). Scale horizontally:
- Each API instance loads its own copy of the bi-encoder (~250 MB) and cross-encoder (~45 MB)
- At 4 Gunicorn workers per instance and 3 instances, you have 12 concurrent request handlers
- For GPU-accelerated inference, assign one GPU per API pod and use `BIENCODER_DEVICE=cuda`

### Elasticsearch Scaling

For the full 8.8M passage corpus:
- Single node with 8 GB heap handles up to ~5 shards of 1.5 GB each
- For higher query throughput, add replica shards (each shard replica handles read requests independently)
- For higher indexing throughput, increase primary shards (3 primary + 1 replica is a common production baseline)

### Qdrant Scaling

Qdrant supports horizontal sharding in its cluster mode:
- Single node: suitable for up to ~10M 768d vectors with INT8 quantization (~7 GB)
- Multi-node cluster: shard the collection across nodes for larger corpora

---

## Database Migrations

PostgreSQL schema migrations are managed by Alembic:

```bash
# Apply all pending migrations
make db-migrate

# Roll back the last migration
make db-rollback

# Create a new migration after schema changes
make db-revision MSG="add experiment tags column"
```

Migrations run automatically on container startup if `ALEMBIC_AUTO_MIGRATE=true` is set in `.env`.

---

## Backup Strategies

### Elasticsearch

```bash
# Register a snapshot repository (local filesystem)
curl -X PUT "localhost:9200/_snapshot/my_backup" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "fs",
    "settings": {
      "location": "/usr/share/elasticsearch/snapshots",
      "compress": true
    }
  }'

# Create a snapshot
curl -X PUT "localhost:9200/_snapshot/my_backup/snapshot_1?wait_for_completion=true"
```

For cloud deployments, use the S3/GCS/Azure repository plugins.

### Qdrant

```bash
# Create a Qdrant snapshot
curl -X POST "localhost:6333/collections/vectorlift_dense/snapshots"

# Download the snapshot
curl -O "localhost:6333/collections/vectorlift_dense/snapshots/snapshot_name.snapshot"

# Restore a snapshot
curl -X POST "localhost:6333/collections/vectorlift_dense/snapshots/upload" \
  --data-binary @snapshot_name.snapshot
```

### PostgreSQL

```bash
# Dump
docker compose exec postgres pg_dump -U vectorlift vectorlift > backup.sql

# Restore
docker compose exec -T postgres psql -U vectorlift vectorlift < backup.sql
```

Automate backups with a Kubernetes CronJob or a Prefect scheduled flow that runs nightly and stores snapshots to object storage (S3, GCS).

### FAISS Index

The FAISS index is a file artifact (`data/indexes/dense/faiss.index`). Back it up like any large file:

```bash
# Copy to S3
aws s3 cp data/indexes/dense/faiss.index s3://your-bucket/vectorlift/faiss.index

# Restore
aws s3 cp s3://your-bucket/vectorlift/faiss.index data/indexes/dense/faiss.index
```

Regenerating the FAISS index from scratch is also an option — the embedding generator has resume capability, so it can be interrupted and restarted.

---

## Health Checks and Monitoring

The `GET /health` endpoint is the canonical health check for all load balancers and Kubernetes liveness probes:

```yaml
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3
```

The API returns `200` with `{"status": "healthy"}` when all backends are reachable and models are loaded, `200` with `{"status": "degraded"}` when some backends are down, and `503` only for catastrophic failures.

Set up Grafana alerts on:
- `vectorlift_http_search_latency_seconds{quantile="0.95"}` > 500ms
- `vectorlift_http_search_total{http_status="500"}` > 10/min
- Elasticsearch cluster health `yellow` or `red`
- Qdrant collections status not `green`
