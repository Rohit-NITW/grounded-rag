# Northstar Engineering — API Platform Guide

This is fictional sample documentation for a software portfolio demonstration.

## Authentication

Clients authenticate using a scoped API token in the Authorization header with the Bearer scheme. API tokens must be stored in the secret manager and rotated every ninety days. A token can be restricted to read-only operations or to a specific project. Never embed a production token in browser JavaScript or commit it to source control. A revoked token stops working immediately.

## Rate limits and retries

The standard API quota is 120 requests per minute per project. Requests above the quota receive HTTP 429 with a Retry-After header. Clients should honor Retry-After and use exponential backoff with jitter. A maximum of three retry attempts is recommended for transient server errors. Do not automatically retry non-idempotent write requests unless an idempotency key is supplied.

## Idempotency

Use the Idempotency-Key header for payment creation requests. The platform stores each idempotency key and its response for twenty-four hours. Repeating the same request with the same key returns the original response. Reusing a key with a different request body returns HTTP 409. Keys must be unique within a project.

## Pagination and versions

List endpoints use cursor pagination with a default page size of twenty and a maximum page size of one hundred. The response includes next_cursor when another page is available. Breaking API changes require a new major version. Deprecated versions receive at least ninety days of notice before removal. The migration guide must include request and response examples.
