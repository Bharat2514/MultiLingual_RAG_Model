---
doc_id: en_open_source_policy
filename: open_source_policy.pdf
language: en
title: Open Source Software Usage and Contribution Policy
---

BHARAT TECHNOLOGIES PRIVATE LIMITED
OPEN SOURCE SOFTWARE USAGE AND CONTRIBUTION POLICY

Document Reference: ENG/OSS/2024/03
Effective Date: 1 June 2024
Owner: Head of Engineering

1. WHY THIS POLICY EXISTS

Open source components make up the majority of every product we ship. That is a deliberate and beneficial choice, but it carries two obligations that are easy to breach by accident: the licence terms of what we consume, and the security posture of code we did not write. This policy sets out how a component is brought in, kept current, and, where we wish, contributed back to.

2. LICENCE CATEGORIES

Permissive licences, being MIT, BSD in its two and three clause forms, Apache 2.0 and ISC, are approved for use in any product without further review.

Weak copyleft licences, being the Mozilla Public License 2.0, the Eclipse Public License and the GNU Lesser General Public License, are approved for use as an unmodified dynamically linked dependency. Modifying such a component, or linking it statically, requires the approval of the Head of Engineering, because it changes what must be disclosed.

Strong copyleft licences, being the GNU General Public License in any version and the Affero General Public License, are not approved for any product distributed to a customer. They may be used for internal tooling that is not distributed.

A component whose licence cannot be determined, or which carries no licence file at all, must not be used. An unlicensed public repository is not public domain.

<!-- page -->

3. INTRODUCING A NEW DEPENDENCY

An engineer adding a direct dependency records the component name, the version, the licence and the reason in the dependency register, which is generated from the manifest files rather than maintained by hand.

Before a new direct dependency is merged, the following must hold: the licence is in an approved category; the component has had a release within the last eighteen months, or its absence of releases is explained in the pull request; there is no known unpatched critical vulnerability; and the component is fetched from the canonical registry, never from a fork or a mirror.

A dependency that pulls in more than fifty transitive packages for a single utility function should be replaced by that function.

4. VULNERABILITY MANAGEMENT

Dependency scanning runs on every pull request and nightly on every default branch. A critical vulnerability must be remediated within seven days of the advisory. A high severity vulnerability must be remediated within thirty days. Medium and low severity findings are remediated in the ordinary course of dependency upgrades.

Where no patched version exists, the engineering lead records a compensating control and a review date. An accepted risk without a review date is not an accepted risk.

<!-- page -->

5. CONTRIBUTING BACK

An employee may contribute a fix or a feature to an open source project used by the company, on company time, with the approval of their engineering lead. Approval is recorded in the pull request that adds the change internally.

A contribution must not include company proprietary code, customer data, internal hostnames, credentials or unreleased product plans. Where a project requires a contributor licence agreement, it is signed by the company rather than by the individual, and the request goes through the legal team.

An employee's personal open source work, done on personal time and personal equipment and unrelated to the company's line of business, is their own and is not covered by this policy.

6. RELEASING OUR OWN CODE AS OPEN SOURCE

A proposal to release company code under an open source licence goes to the Head of Engineering and the legal team. The default licence for a company release is Apache 2.0, which grants patent rights explicitly.

Before release, the repository history is examined for credentials and customer data, because history is published together with the current state, and rewriting it after publication does not recall what was already fetched.

7. ATTRIBUTION

Every product that is distributed to a customer ships a notices file listing each open source component, its version, its licence and the required copyright notice. The notices file is generated as part of the build, so that a dependency added after the last manual review cannot be omitted.

8. ENFORCEMENT

A pull request that introduces a component in an unapproved licence category fails the build. The check may be overridden only by the Head of Engineering, and every override is recorded with its justification and reviewed quarterly.
