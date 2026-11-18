# Northstar Engineering — Observability Standards

This is fictional sample documentation for a software portfolio demonstration.

## Service objectives

The customer API has a monthly availability objective of 99.9 percent. The latency objective is a p95 response time below 500 milliseconds over a rolling one-hour window. Health checks alone do not prove that the customer workflow is healthy; use synthetic checks that create and retrieve a test resource. The service owner reviews error-budget consumption every week.

## Metrics

Every service reports request count, error count, and latency histograms. Metrics use service, environment, and route-template labels. Never place customer identifiers, document names, or raw URLs into metric labels because they create high cardinality and may expose private information. Batch jobs also report completion time and failed-item count.

## Tracing and logging

Propagate a request ID across API boundaries and include it in every structured log entry. Trace ten percent of successful requests and all failed requests when the tracing backend supports tail sampling. Redact authorization headers and customer document content before exporting logs or traces. Logs should describe the operation, outcome, and duration, without storing the request body.

## Alert quality

Page the on-call engineer when the error rate exceeds five percent for five consecutive minutes. Each paging alert must link a runbook, identify an owner, and describe customer impact. Non-urgent capacity warnings go to the weekly reliability queue. Review noisy alerts after every incident and disable duplicate alerts once an equivalent signal has been verified. The primary on-call engineer follows the escalation timings in the incident response guide.
