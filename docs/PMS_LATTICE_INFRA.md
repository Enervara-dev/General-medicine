# PMS integration — VPC Lattice infrastructure requirements

Status: **NOT IMPLEMENTED.** This document specifies the AWS infrastructure the
GM→PMS SigV4 integration depends on. The GM application code is complete and
merged into the working tree; it cannot reach PMS until the resources below
exist.

No Terraform was authored for this. This repository contains no `.tf` files and
no infrastructure module, so there was nothing to extend and no existing naming,
tagging, or module convention to match. Authoring greenfield Terraform against
invented VPC / ALB / role identifiers risked creating duplicate infrastructure —
explicitly out of scope. This spec is the hand-off instead.

## Target path

```
GM ECS task
  └─ SigV4 (service=vpc-lattice-svcs, region=ap-south-1), ECS task-role creds
     └─ VPC Lattice service  [auth_type = AWS_IAM]
        └─ Lattice target group  [type = ALB]
           └─ existing internal PMS ALB      ← reuse, do not recreate
              └─ PMS ECS
```

## Resources required

### 1. Lattice service network

- One service network, associated with the VPC that already hosts GM and PMS.
- `auth_type = "AWS_IAM"`.
- VPC association must include the security group that permits GM's task ENIs to
  reach the Lattice data plane.

### 2. Lattice service

- `auth_type = "AWS_IAM"`.
- Associated with the service network above.
- Custom domain optional. If omitted, Lattice issues a generated DNS name that
  becomes `PMS_BASE_URL` — it must be reachable over **HTTPS**.

### 3. Target group + listener

- Target group `type = "ALB"`, targeting the **existing** internal PMS ALB ARN.
  Do not create a new ALB or ECS service.
- Note the AWS constraint: ALB-type Lattice target groups do **not** perform
  health checks; Lattice relies on the ALB's own target health.
- Listener on port 443, protocol HTTPS, default action forwarding to the target
  group.

### 4. PMS ALB security group

Currently the PMS ALB SG admits the broad VPC CIDR. Replace that ingress rule
with one that admits **only** the Lattice-managed prefix list or the security
group used for the Lattice VPC association. This is what stops any workload in
the VPC bypassing Lattice — and therefore bypassing IAM authentication — by
calling the ALB directly. Without this change, SigV4 is advisory rather than
enforced.

### 5. Lattice auth policy

Attached to the Lattice service. Allow only GM's ECS **task role** to invoke:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "AWS": "arn:aws:iam::341208406542:role/<GM_ECS_TASK_ROLE>" },
    "Action": "vpc-lattice-svcs:Invoke",
    "Resource": "*"
  }]
}
```

Deny-by-default is implicit: any principal not listed is rejected by Lattice
before the request reaches the ALB.

### 6. GM ECS task-role IAM policy

Add to the task role (**not** the execution role — the task role is the identity
application code assumes via the container credential provider):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "vpc-lattice-svcs:Invoke",
    "Resource": "arn:aws:vpc-lattice:ap-south-1:341208406542:service/<SERVICE_ID>"
  }]
}
```

Scope `Resource` to the specific service ARN rather than `*`.

## GM environment variables to set on the ECS task definition

| Variable | Value | Notes |
|---|---|---|
| `PMS_BASE_URL` | `https://<lattice-service-dns>` | **Must be https.** A plaintext URL is refused and the client degrades to `NullPMSClient`. |
| `ENABLE_PMS_SHADOW` | `true` | Master switch. |
| `PMS_SIGV4_ENABLED` | `true` | Default. Set `false` only for a local stub. |
| `PMS_SIGV4_SERVICE` | `vpc-lattice-svcs` | Must match the fronting component. |
| `PMS_SIGV4_REGION` | `ap-south-1` | Must match the region the Lattice service lives in. |
| `PMS_INGEST_PATH` | *unconfirmed* | See blocker below. |

The deploy workflow reads the live task definition and swaps only the image, so
these persist across deploys once set. No secret is involved — SigV4 uses the
task role, so there is nothing to put in Secrets Manager.

## Open PMS-side blockers

1. **Canonical ingest path.** Code and contract say
   `/v1/clinical-memory/events`; the SigV4 brief said `/v1/memory/events`.
   Unresolved — left as-is in code, config-driven via `PMS_INGEST_PATH`.
2. **Consumer identity derivation.** PMS must read the caller ARN from the
   Lattice-injected request context and map it to `general_medicine` via an
   explicit allowlist. GM no longer sends `X-Consumer-Id`.
3. **Patient-scope authorization.** Unenforced. See `app/services/pms/assertions.py`.
4. **Response contract.** Status codes, error body shape, and the
   `Idempotency-Key` dedup window are still undefined by PMS.
