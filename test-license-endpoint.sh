#!/bin/bash

# Test the license-check endpoint locally

SERVER="http://localhost:8000"

echo "Testing license-check endpoint..."
echo ""

# Test 1: Apache license (should return true)
echo "Test 1: google-research/bert (Apache)"
RESPONSE=$(curl -s -X POST "$SERVER/artifact/model/9430210313/license-check" \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/google-research/bert"}')
echo "Response: $RESPONSE"
echo "Response length: ${#RESPONSE}"
echo "Hex dump:"
echo -n "$RESPONSE" | od -A x -t x1z -v
echo ""

# Test 2: LGPL license (should return false)
echo "Test 2: microsoft/git (LGPL)"
RESPONSE=$(curl -s -X POST "$SERVER/artifact/model/695465474/license-check" \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/microsoft/git"}')
echo "Response: $RESPONSE"
echo "Response length: ${#RESPONSE}"
echo "Hex dump:"
echo -n "$RESPONSE" | od -A x -t x1z -v
echo ""

# Test 3: Invalid artifact ID (should return 404)
echo "Test 3: invalidId (should 404)"
curl -s -X POST "$SERVER/artifact/model/invalidId/license-check" \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/google-research/bert"}' \
  -w "\nHTTP Status: %{http_code}\n"
