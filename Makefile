# Makefile for MedExplain-GNN

# Variables
NAMESPACE = medexplain-gnn
K8S_FILES = k8s/

# Local Docker Compose
local-up:
	./setup.sh

local-down:
	docker-compose down

# Kubernetes Deployment
k8s-deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/neo4j.yaml
	kubectl apply -f k8s/mongodb.yaml
	kubectl apply -f k8s/redis.yaml
	# Wait for DBs to be ready before deploying AI
	sleep 10
	kubectl apply -f k8s/ai-service.yaml
	kubectl apply -f k8s/backend.yaml
	kubectl apply -f k8s/frontend.yaml

k8s-status:
	kubectl get pods -n $(NAMESPACE)

k8s-populate:
	@echo "--- Populating Graph ---"
	$(eval AI_POD := $(shell kubectl get pods -n $(NAMESPACE) -l app=ai-service -o jsonpath='{.items[0].metadata.name}'))
	kubectl exec -it $(AI_POD) -n $(NAMESPACE) -- python populate_graph.py

k8s-port-forward:
	kubectl port-forward svc/frontend 3000:3000 -n $(NAMESPACE) &
	kubectl port-forward svc/backend 8000:8000 -n $(NAMESPACE) &
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000/docs"

k8s-clean:
	kubectl delete namespace $(NAMESPACE)

.PHONY: local-up local-down k8s-deploy k8s-status k8s-populate k8s-port-forward k8s-clean
