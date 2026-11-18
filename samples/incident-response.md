# Northstar Engineering — Incident Response

This is fictional sample documentation for a software portfolio demonstration.

## Severity and ownership

A SEV1 incident means complete production unavailability or confirmed exposure of customer data. A SEV2 incident means a major feature is unavailable for a subset of customers. A SEV3 incident is a minor defect with a documented workaround. Only the incident commander can downgrade an active incident after checking the customer impact.

## On-call escalation

The primary on-call engineer must acknowledge a production page within five minutes. If the primary does not acknowledge within five minutes, page the secondary on-call engineer. If no engineer acknowledges within ten minutes of the first page, escalate to the engineering manager. The first responding engineer acts as incident commander until ownership is explicitly handed over.

## Communication

For a SEV1 incident, publish a status update every fifteen minutes until recovery. Every update must state the customer impact, current mitigation, and next update time. Do not include customer records, access tokens, or private debugging output in status updates. The incident scribe maintains a timeline of decisions in the incident ticket.

## Recovery and review

An incident is resolved after the customer-facing service has been healthy for thirty consecutive minutes. A blameless postmortem is due within three business days of resolution for every SEV1 and SEV2 incident. The postmortem must include contributing factors, detection gaps, and action items with owners and due dates. Action items are reviewed in the next weekly reliability meeting. Deployment-related incidents should link the release ticket and the exact image digest used for recovery.
