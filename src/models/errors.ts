export class AppError extends Error {
  public readonly code: string;

  public constructor(code: string, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
    this.code = code;
  }
}

export class ConfigError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("CONFIG_ERROR", message, options);
  }
}

export class ValidationError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("VALIDATION_ERROR", message, options);
  }
}

export class ContractError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("CONTRACT_ERROR", message, options);
  }
}
