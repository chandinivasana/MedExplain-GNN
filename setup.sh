#!/bin/bash
set -e

# MedExplain-GNN Setup Script

echo "--- Building and starting microservices with Docker ---"
docker-compose up --build -d || { echo "Docker Compose failed to start. Check logs for details."; exit 1; }

echo "--- Waiting for Neo4j to start... ---"
until curl -s http://localhost:7474 > /dev/null; do
  echo "Neo4j is not ready yet... sleeping for 5s"
  sleep 5
done

echo "--- Populating the Knowledge Graph ---"
# Run population inside the container to ensure drivers are present
docker exec ai-service python database/populate_graph.py

echo "--- Setup complete! ---"
echo "Frontend: http://localhost:3000"
echo "Backend Gateway: http://localhost:8000"
echo "AI Service: http://localhost:8001"
echo "Neo4j Console: http://localhost:7474 (user: neo4j, pass: password)"
