# Northstar Engineering — Data Protection and Backups

This is fictional sample documentation for a software portfolio demonstration.

## Backups

Production database backups run every six hours. Backups are encrypted and retained for thirty days. A restore test is performed every month in an isolated environment with no access to production credentials. The recovery point objective is six hours and the recovery time objective is two hours. The database owner records the actual restore duration and any failures after every restore test.

## Retention

Application logs are retained for fourteen days. Security audit logs are retained for one year. Logs must not contain passwords, API keys, payment card data, or raw customer document content. Debug logging is disabled by default in production. Temporary diagnostic logging requires a ticket with an expiry time and an assigned reviewer.

## Customer deletion

After a verified customer deletion request, active customer records must be removed within seven days. Deleted records in existing backups expire through the standard thirty-day backup retention schedule. The support team records the request identifier and completion date without copying the deleted content into the ticket. The data owner checks dependent search indexes and caches as part of the deletion workflow.

## Access review

Production database access is limited to named accounts with multi-factor authentication. Shared database accounts are prohibited. Access reviews happen quarterly and require each team lead to confirm that every grant is still necessary. Temporary emergency access expires after four hours unless the incident commander renews it. All emergency access changes must reference an incident ticket.
