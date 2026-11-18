# Northstar Engineering — Deployment Runbook

This is fictional sample documentation for a software portfolio demonstration.

## Release process

Production deployments happen on weekdays between 09:00 and 16:00 UTC. The release owner must confirm that unit tests, integration tests, and database migration checks pass before promoting a release. Each release requires approval from one engineer who did not author the change. A change that modifies authentication requires an additional security review. Release artifacts are immutable container images identified by a Git commit SHA.

The deployment pipeline first sends five percent of traffic to the canary for ten minutes. The release owner compares the canary error rate and p95 latency with the stable version. Promote the canary only if the error rate stays below one percent and p95 latency remains below 500 milliseconds. Record the image digest and the release ticket in the deployment log.

## Rollback procedure

To roll back production, pause the deployment pipeline, select the previous healthy image digest, and redeploy that image. Verify the health endpoint and error dashboard for ten minutes after the rollback. Notify the on-call engineer and attach the failed release's logs to the incident ticket. Never reverse a destructive database migration automatically; follow the migration recovery plan approved for that change.

## Freeze periods

No routine releases are allowed during the monthly billing run. Emergency fixes require the incident commander's approval and the same post-deployment verification as a normal release. The release owner is responsible for updating the release notes before closing the release ticket.
