#!/bin/bash

# TechMart Customer Service Demo - Build and Push All Container Images
# This script builds and pushes all container images required for the demo

# Note: We don't use 'set -e' here because we want to continue building
# all images even if one fails, and report the summary at the end

# Resolve the demo root from this script's own location so the script works
# from any working directory. Uses $0 rather than BASH_SOURCE so it behaves
# the same whether invoked as 'bash ...' or 'sh ...'.
DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="${CONTAINER_REGISTRY:-quay.io}"

if [ -z "${REGISTRY_USER}" ]; then
    echo -e "${RED}Error: REGISTRY_USER is not set.${NC}"
    echo "Please export your registry username before running this script:"
    echo "  export REGISTRY_USER=your-quay-username"
    exit 1
fi
MCP_TAG="${MCP_TAG:-mcp-server}"
UI_TAG="${UI_TAG:-ui}"
DB_INIT_TAG="${DB_INIT_TAG:-db-init}"
# Image names
UI_IMAGE="${REGISTRY}/${REGISTRY_USER}/techmart:${UI_TAG}"
MCP_IMAGE="${REGISTRY}/${REGISTRY_USER}/techmart:${MCP_TAG}"
DB_INIT_IMAGE="${REGISTRY}/${REGISTRY_USER}/techmart:${DB_INIT_TAG}"

# Optional single-image selector: ui | mcp | db-init
TARGET="${1:-all}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TechMart Container Image Builder${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Registry: ${REGISTRY}"
echo -e "  User: ${REGISTRY_USER}"
echo -e "  Target:   ${TARGET}"

echo ""
echo -e "${YELLOW}Images to build:${NC}"
echo -e "  1. UI: ${UI_IMAGE}"
echo -e "  2. MCP Server: ${MCP_IMAGE}"
echo -e "  3. DB Init: ${DB_INIT_IMAGE}"
echo ""

# Function to build and push an image
build_and_push() {
    local name=$1
    local dockerfile=$2
    local context=$3
    local image=$4
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Building ${name}...${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ ! -f "${dockerfile}" ]; then
        echo -e "${RED}Error: Containerfile not found: ${dockerfile}${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Building image: ${image}${NC}"
    podman build --network host -f "${dockerfile}" -t "${image}" "${context}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful${NC}"
        echo ""
        echo -e "${YELLOW}Pushing image to registry...${NC}"
        podman push "${image}"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Push successful${NC}"
            echo ""
            return 0
        else
            echo -e "${RED}✗ Push failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Build failed${NC}"
        return 1
    fi
}

# Check if podman is installed
if ! command -v podman &> /dev/null; then
    echo -e "${RED}Error: podman is not installed${NC}"
    echo "Please install podman first"
    exit 1
fi

# Check if logged in to registry
echo -e "${YELLOW}Checking registry authentication...${NC}"
if ! podman login ${REGISTRY} --get-login &> /dev/null; then
    echo -e "${YELLOW}Not logged in to ${REGISTRY}${NC}"
    echo -e "${YELLOW}Please log in:${NC}"
    podman login ${REGISTRY}
    if [ $? -ne 0 ]; then
        echo -e "${RED}Login failed. Exiting.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ Authenticated to ${REGISTRY}${NC}"
echo ""

# Build and push images
SUCCESS_COUNT=0
FAIL_COUNT=0

build_image() {
    local name=$1; local dockerfile=$2; local image=$3
    # Containerfile paths and the build context are both anchored to the demo
    # root, so COPY paths inside the Containerfiles resolve the same way no
    # matter where the script is invoked from.
    if build_and_push "${name}" "${DEMO_DIR}/${dockerfile}" "${DEMO_DIR}" "${image}"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
}

case "${TARGET}" in
  ui)      build_image "UI Application"      "docker/Containerfile.ui"      "${UI_IMAGE}" ;;
  mcp)     build_image "MCP Server"          "docker/Containerfile.mcp"     "${MCP_IMAGE}" ;;
  db-init) build_image "Database Initializer" "docker/Containerfile.db-init" "${DB_INIT_IMAGE}" ;;
  all)
    build_image "UI Application"       "docker/Containerfile.ui"      "${UI_IMAGE}"
    build_image "MCP Server"           "docker/Containerfile.mcp"     "${MCP_IMAGE}"
    build_image "Database Initializer" "docker/Containerfile.db-init" "${DB_INIT_IMAGE}"
    ;;
  *)
    echo -e "${RED}Unknown target: ${TARGET}${NC}"
    echo "Usage: $0 [ui|mcp|db-init|all]"
    exit 1
    ;;
esac

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Build Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}Failed: ${FAIL_COUNT}${NC}"
echo ""

if [ ${FAIL_COUNT} -eq 0 ]; then
    echo -e "${GREEN}✓ All images built and pushed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Update image references in deployment YAMLs if needed"
    echo -e "  2. Deploy to OpenShift: bash scripts/manage-resources.sh deploy --all"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some images failed to build/push${NC}"
    echo -e "${YELLOW}Please check the errors above and try again${NC}"
    exit 1
fi
