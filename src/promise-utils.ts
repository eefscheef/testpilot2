function sleep(milliseconds: number) {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

export interface RetryOptions {
  shouldRetry?: (error: unknown, attempt: number) => boolean;
  getDelayMs?: (error: unknown, attempt: number) => number;
}

/**
 * This function provides supports for retrying the creation of a promise
 * up to a given number of times in case the promise is rejected.
 * This is useful for, e.g., retrying a request to a server that is temporarily unavailable.
 *
 */
export async function retry<T>(
  f: () => Promise<T>,
  howManyTimes: number,
  options: RetryOptions = {}
): Promise<T> {
  const totalAttempts = Math.max(1, howManyTimes);
  for (let attempt = 1; attempt <= totalAttempts; attempt++) {
    try {
      if (attempt > 1) {
        console.log(`  retry ${attempt}/${totalAttempts}`);
      }
      return await f();
    } catch (error) {
      const isLastAttempt = attempt === totalAttempts;
      const shouldRetry =
        !isLastAttempt && (options.shouldRetry?.(error, attempt) ?? true);
      console.log(`Promise rejected with ${error}.`);
      if (!shouldRetry) {
        throw error;
      }

      const delayMs = Math.max(0, options.getDelayMs?.(error, attempt) ?? 0);
      if (delayMs > 0) {
        console.log(`  waiting ${delayMs} ms before retry`);
        await sleep(delayMs);
      }
    }
  }
  throw new Error("retry exhausted unexpectedly");
}

export interface IRateLimitMetadata {
  estimatedTokens?: number;
}

/**
 * This interface provides supports for retrying the creation of a promise
 */
export interface IRateLimiter {
  /**
   * Waits until the rate limiter allows the next request, then evaluate the function that
   * produces the promise
   */
  next<T>(p: () => Promise<T>, metadata?: IRateLimitMetadata): Promise<T>;

  /**
   * returns a description of the rate limiter
   */
  getDescription(): string;
}

/**
 * This class provides supports for asynchronous rate limiting by
 * limiting the number of requests to the server to at most one
 * in N milliseconds. This is useful for throttling requests to
 * a server that has a limit on the number of requests per second.
 */
export abstract class RateLimiter implements IRateLimiter {
  constructor(protected howManyMilliSeconds: number) {
    this.timer = this.resetTimer();
  }
  /**
   * the timer is a promise that is resolved after a certain number of milliseconds
   * have elapsed. The timer is reset after each request.
   */
  private timer: Promise<void>;

  /**
   *  Waits until the timer has expired, then evaluate the function that
   * produces the promise
   * @param p a function that produces a promise
   * @returns returns the promise produced by the function p (after the timer has expired)
   */
  public async next<T>(
    p: () => Promise<T>,
    _metadata?: IRateLimitMetadata
  ): Promise<T> {
    await this.timer; // wait until timer has expired
    this.timer = this.resetTimer(); // reset timer (for the next request)
    return p(); // return the promise
  }

  public abstract getDescription(): string;

  /**
   * resets the timer
   * @returns a promise that is resolved after the number of milliseconds
   *         specified in the constructor have elapsed
   */
  protected resetTimer = () =>
    new Promise<void>((resolve, reject) => {
      setTimeout(() => {
        resolve();
      }, this.howManyMilliSeconds);
    });
}

/**
 * A rate limiter that limits the number of requests to the server to a
 * maximum of one per N milliseconds.
 *
 */
export class FixedRateLimiter extends RateLimiter implements IRateLimiter {
  public constructor(N: number) {
    super(N);
  }

  /**
   * returns a description of the rate limiter
   */
  public getDescription(): string {
    return `FixedRateLimiter (1 request per ${this.howManyMilliSeconds} ms)`;
  }
}

/**
 * A custom rate limiter for use during benchmark runs. It increases
 * the pace of requests after two designated thresholds have been reached.
 */
export class BenchmarkRateLimiter extends RateLimiter {
  private requestCount: number;

  private static INITIAL_PACE = 10000;
  private static PACE_AFTER_150_REQUESTS = 5000;
  private static PACE_AFTER_300_REQUESTS = 2500;

  constructor() {
    console.log(
      `BenchmarkRateLimiter: initial pace is ${BenchmarkRateLimiter.INITIAL_PACE}`
    );
    super(BenchmarkRateLimiter.INITIAL_PACE);
    this.requestCount = 0;
  }

  public next<T>(
    p: () => Promise<T>,
    _metadata?: IRateLimitMetadata
  ): Promise<T> {
    this.requestCount++;
    if (this.requestCount === 150) {
      this.howManyMilliSeconds = BenchmarkRateLimiter.PACE_AFTER_150_REQUESTS;
      console.log(
        `BenchmarkRateLimiter: increasing pace to ${BenchmarkRateLimiter.PACE_AFTER_150_REQUESTS}`
      );
    } else if (this.requestCount === 300) {
      this.howManyMilliSeconds = BenchmarkRateLimiter.PACE_AFTER_300_REQUESTS;
      console.log(
        `BenchmarkRateLimiter: increasing pace to ${BenchmarkRateLimiter.PACE_AFTER_300_REQUESTS}`
      );
    }
    return super.next(p);
  }

  /**
   * returns a description of the rate limiter
   */
  public getDescription(): string {
    return `BenchmarkRateLimiter (increasing pace after 150 and 300 requests)`;
  }
}

/**
 * A rate limiter that does not limit the rate of requests to the server.
 */
export class NoRateLimiter implements IRateLimiter {
  public async next<T>(
    p: () => Promise<T>,
    _metadata?: IRateLimitMetadata
  ): Promise<T> {
    return p();
  }

  /**
   * returns a description of the rate limiter
   */
  public getDescription(): string {
    return `NoRateLimiter`;
  }
}

/**
 * A rate limiter that caps the estimated number of tokens spent over a sliding
 * 60 second window.
 */
export class TokenRateLimiter implements IRateLimiter {
  private readonly reservations: { timestamp: number; tokens: number }[] = [];

  public constructor(private readonly tokensPerMinute: number) {}

  public async next<T>(
    p: () => Promise<T>,
    metadata: IRateLimitMetadata = {}
  ): Promise<T> {
    const estimatedTokens = Math.max(
      1,
      Math.ceil(metadata.estimatedTokens ?? 1)
    );
    if (estimatedTokens > this.tokensPerMinute) {
      console.warn(
        `TokenRateLimiter: single request estimate ${estimatedTokens} exceeds configured limit ${this.tokensPerMinute}; proceeding anyway.`
      );
    }

    while (true) {
      const now = Date.now();
      this.prune(now);
      const usedTokens = this.reservations.reduce(
        (sum, reservation) => sum + reservation.tokens,
        0
      );
      const reservedTokens = Math.min(estimatedTokens, this.tokensPerMinute);
      if (usedTokens + reservedTokens <= this.tokensPerMinute) {
        this.reservations.push({ timestamp: now, tokens: reservedTokens });
        return p();
      }

      const oldest = this.reservations[0];
      const waitMs = Math.max(1, 60000 - (now - oldest.timestamp));
      console.log(
        `TokenRateLimiter: waiting ${waitMs} ms to stay under ${this.tokensPerMinute} estimated tokens/minute`
      );
      await sleep(waitMs);
    }
  }

  public getDescription(): string {
    return `TokenRateLimiter (${this.tokensPerMinute} estimated tokens per minute)`;
  }

  private prune(now: number) {
    while (
      this.reservations.length > 0 &&
      now - this.reservations[0].timestamp >= 60000
    ) {
      this.reservations.shift();
    }
  }
}

export class CompositeRateLimiter implements IRateLimiter {
  public constructor(private readonly limiters: IRateLimiter[]) {}

  public next<T>(
    p: () => Promise<T>,
    metadata: IRateLimitMetadata = {}
  ): Promise<T> {
    const applyLimiter = (index: number): Promise<T> => {
      if (index >= this.limiters.length) {
        return p();
      }
      return this.limiters[index].next(() => applyLimiter(index + 1), metadata);
    };

    return applyLimiter(0);
  }

  public getDescription(): string {
    return this.limiters.map((limiter) => limiter.getDescription()).join(" + ");
  }
}
