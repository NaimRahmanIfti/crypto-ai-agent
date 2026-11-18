# Reference Implementation

**siwe.ts**

```typescript
/**
 * Sign-In with Ethereum (SIWE) implementation using viem
 */
import { verifyMessage, type Address, type Hex, type PublicClient } from "viem";
import {
  createSiweMessage,
  generateSiweNonce,
  parseSiweMessage,
  validateSiweMessage,
  type SiweMessage,
} from "viem/siwe";

// ============================================================================
// Types
// ============================================================================

export interface SiweSessionData {
  address: Address;
  chainId: number;
  domain: string;
  nonce: string;
  issuedAt: string;
  expirationTime?: string;
  notBefore?: string;
  requestId?: string;
  resources?: string[];
}

export interface SiweVerifyParams {
  message: string;
  signature: Hex;
  expectedDomain?: string;
  expectedChainId?: number;
}

export interface SiweSession {
  address: Address;
  chainId: number;
  isValid: boolean;
  expiresAt?: Date;
  data: SiweSessionData;
}

export interface NonceResult {
  nonce: string;
  issuedAt: string;
  expiresAt: string;
}

// ============================================================================
// Nonce Management
// ============================================================================

// In-memory nonce store (replace with Redis/database in production)
const nonceStore = new Map<string, { nonce: string; createdAt: number }>();
const NONCE_TTL = 5 * 60 * 1000; // 5 minutes

/**
 * Generate a new SIWE nonce
 */
export function generateNonce(): NonceResult {
  const nonce = generateSiweNonce();
  const issuedAt = new Date().toISOString();
  const expiresAt = new Date(Date.now() + NONCE_TTL).toISOString();

  // Store nonce
  nonceStore.set(nonce, { nonce, createdAt: Date.now() });

  // Clean up expired nonces
  cleanupExpiredNonces();

  return { nonce, issuedAt, expiresAt };
}

/**
 * Verify a nonce is valid and unused
 */
export function verifyNonce(nonce: string): boolean {
  const stored = nonceStore.get(nonce);
  if (!stored) return false;

  // Check if expired
  if (Date.now() - stored.createdAt > NONCE_TTL) {
    nonceStore.delete(nonce);
    return false;
  }

  return true;
}

/**
 * Consume a nonce (mark as used)
 */
export function consumeNonce(nonce: string): boolean {
  if (!verifyNonce(nonce)) return false;
  nonceStore.delete(nonce);
  return true;
}

/**
 * Clean up expired nonces
 */
function cleanupExpiredNonces(): void {
  const now = Date.now();
  for (const [nonce, data] of nonceStore.entries()) {
    if (now - data.createdAt > NONCE_TTL) {
      nonceStore.delete(nonce);
    }
  }
}

// ============================================================================
// Message Creation
// ============================================================================

export interface CreateMessageParams {
  address: Address;
  chainId: number;
  domain: string;
  uri: string;
  nonce?: string;
  statement?: string;
  expirationTime?: Date;
  notBefore?: Date;
  requestId?: string;
  resources?: string[];
}

/**
 * Create a SIWE message
 */
export function createMessage(params: CreateMessageParams): string {
  const nonce = params.nonce ?? generateSiweNonce();

  return createSiweMessage({
    address: params.address,
    chainId: params.chainId,
    domain: params.domain,
    uri: params.uri,
    nonce,
    version: "1",
    statement: params.statement,
    expirationTime: params.expirationTime,
    notBefore: params.notBefore,
    requestId: params.requestId,
    resources: params.resources,
  });
}

// ============================================================================
// Message Parsing
// ============================================================================

/**
 * Parse a SIWE message string into components
 */
export function parseMessage(message: string): SiweMessage | null {
  try {
    return parseSiweMessage(message) as SiweMessage;
  } catch {
    return null;
  }
}

// ============================================================================
// Signature Verification
// ============================================================================

/**
 * Verify a SIWE signature
 */
export async function verifySiweSignature(
  params: SiweVerifyParams,
  client?: PublicClient,
): Promise<SiweSession> {
  const { message, signature, expectedDomain, expectedChainId } = params;

  // Parse the message
  const parsed = parseMessage(message);
  if (!parsed) {
    return {
      address: "0x0" as Address,
      chainId: 0,
      isValid: false,
      data: {} as SiweSessionData,
    };
  }

  // Validate message fields
  const isMessageValid =
    parsed.address &&
    validateSiweMessage({
      address: parsed.address,
      message: parsed as SiweMessage,
    });

  if (!isMessageValid) {
    return {
      address: parsed.address ?? ("0x0" as Address),
      chainId: parsed.chainId ?? 0,
      isValid: false,
      data: extractSessionData(parsed),
    };
  }

  // Verify domain if specified
  if (expectedDomain && parsed.domain !== expectedDomain) {
    return {
      address: parsed.address ?? ("0x0" as Address),
      chainId: parsed.chainId ?? 0,
      isValid: false,
      data: extractSessionData(parsed),
    };
  }

  // Verify chain ID if specified
  if (expectedChainId && parsed.chainId !== expectedChainId) {
    return {
      address: parsed.address ?? ("0x0" as Address),
      chainId: parsed.chainId ?? 0,
      isValid: false,
      data: extractSessionData(parsed),
    };
  }

  // Verify the nonce
  if (parsed.nonce && !consumeNonce(parsed.nonce)) {
    return {
      address: parsed.address ?? ("0x0" as Address),
      chainId: parsed.chainId ?? 0,
      isValid: false,
      data: extractSessionData(parsed),
    };
  }

  // Verify the signature
  let isSignatureValid = false;

  if (client) {
    // Use public client for smart contract wallet verification (ERC-1271)
    isSignatureValid = await client.verifyMessage({
      address: parsed.address!,
      message,
      signature,
    });
  } else {
    // Basic signature verification for EOA
    isSignatureValid = await verifyMessage({
      address: parsed.address!,
      message,
      signature,
    });
  }

  // Calculate expiration
  let expiresAt: Date | undefined;
  if (parsed.expirationTime) {
    expiresAt = new Date(parsed.expirationTime);
  }

  return {
    address: parsed.address!,
    chainId: parsed.chainId!,
    isValid: isSignatureValid,
    expiresAt,
    data: extractSessionData(parsed),
  };
}

/**
 * Extract session data from parsed message
 */
function extractSessionData(parsed: SiweMessage): SiweSessionData {
  return {
    address: parsed.address ?? ("0x0" as Address),
    chainId: parsed.chainId ?? 0,
    domain: parsed.domain ?? "",
    nonce: parsed.nonce ?? "",
    issuedAt: parsed.issuedAt?.toISOString() ?? new Date().toISOString(),
    expirationTime: parsed.expirationTime?.toISOString(),
    notBefore: parsed.notBefore?.toISOString(),
    requestId: parsed.requestId,
    resources: parsed.resources,
  };
}

// ============================================================================
// Session Management
// ============================================================================

// In-memory session store (replace with Redis/database in production)
const sessionStore = new Map<string, SiweSession>();

/**
 * Create a session from verified SIWE
 */
export function createSession(session: SiweSession): string {
  // Generate session ID
  const sessionId = crypto.randomUUID();

  // Store session
  sessionStore.set(sessionId, session);

  return sessionId;
}

/**
 * Get session by ID
 */
export function getSession(sessionId: string): SiweSession | null {
  const session = sessionStore.get(sessionId);
  if (!session) return null;

  // Check expiration
  if (session.expiresAt && new Date() > session.expiresAt) {
    sessionStore.delete(sessionId);
    return null;
  }

  return session;
}

/**
 * Delete session (logout)
 */
export function deleteSession(sessionId: string): boolean {
  return sessionStore.delete(sessionId);
}

/**
 * Clear all sessions for an address
 */
export function clearSessionsForAddress(address: Address): number {
  let cleared = 0;
  for (const [id, session] of sessionStore.entries()) {
    if (session.address.toLowerCase() === address.toLowerCase()) {
      sessionStore.delete(id);
      cleared++;
    }
  }
  return cleared;
}
```

**index.ts**

```typescript
/**
 * Sign-In with Ethereum implementation for secure authentication
 */

// Export all SIWE functions
export * from "./siwe";

// ============================================================================
// Auth Module API
// ============================================================================

import type { PublicClient } from "viem";
import {
  generateNonce as _generateNonce,
  verifySiweSignature,
  createMessage as _createMessage,
  createSession,
  getSession,
  deleteSession,
  type SiweSession,
  type NonceResult,
  type CreateMessageParams,
  type SiweVerifyParams,
} from "./siwe";

/**
 * Auth module providing SIWE authentication
 */
export const auth = {
  /**
   * Generate a new nonce for SIWE
   */
  nonce(): NonceResult {
    return _generateNonce();
  },

  /**
   * Create a SIWE message for signing
   */
  createMessage(params: CreateMessageParams): string {
    return _createMessage(params);
  },

  /**
   * Verify a signed SIWE message
   */
  async verify(
    params: SiweVerifyParams,
    client?: PublicClient,
  ): Promise<{ session: SiweSession; sessionId?: string }> {
    const session = await verifySiweSignature(params, client);

    if (session.isValid) {
      const sessionId = createSession(session);
      return { session, sessionId };
    }

    return { session };
  },

  /**
   * Get current session info
   */
  me(sessionId: string): SiweSession | null {
    return getSession(sessionId);
  },

  /**
   * Logout (delete session)
   */
  logout(sessionId: string): boolean {
    return deleteSession(sessionId);
  },
};

// ============================================================================
// Types Re-export
// ============================================================================

export type {
  SiweSession,
  SiweSessionData,
  SiweVerifyParams,
  CreateMessageParams,
  NonceResult,
} from "./siwe";
```
