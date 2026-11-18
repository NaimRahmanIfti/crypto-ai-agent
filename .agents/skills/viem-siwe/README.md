# Viem SIWE Skills

This skill provides a complete, production-ready implementation of Sign-In with Ethereum (SIWE) using `viem` v2.

## Installation

To add this skill to your project, run:

```bash
npx skills add melonask/viem-siwe-skills
```

## Features

- **EIP-4361 Compliant**: Full support for the SIWE standard.
- **Nonce Management**: Secure generation and consumption of nonces to prevent replay attacks.
- **ERC-1271 Support**: Verified signature checks for both EOAs and Smart Contract Wallets.
- **Session Handling**: Patterns for creating and managing authenticated sessions.

## Structure

- `SKILL.md`: Main entry point with usage triggers and high-level guidance.
- `references/implementation.md`: The core auth module implementation (`siwe.ts` and `index.ts`).
- `references/api-docs.md`: Reference documentation for `viem` SIWE utilities.

## Security Best Practices Included

1. **Strict Nonce Verification**: Fails authentication if the nonce is missing, expired, or already used.
2. **Domain/Chain ID Binding**: Ensures signatures are only valid for the intended host and network.
3. **Time-based Validation**: Respects `issuedAt`, `notBefore`, and `expirationTime`.
