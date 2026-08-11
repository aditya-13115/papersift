
# Backend Engineering Learning

A structured collection of backend engineering concepts and hands-on
implementations while learning Python, FastAPI, Pydantic, asynchronous
programming, and backend reliability patterns.

The repository is organized by concept rather than by project, so each
folder represents a specific area of backend engineering.

---

## Repository Structure

```text
backend/
│
├── async_patterns/
│   └── semaphore_timeout.py
│
├── fastapi/
│   ├── async_fastapi.py
│   └── dependency_injection.py
│
├── pydantic/
│   ├── pydantic_models.py
│   ├── nested_models.py
│   ├── pydantic_validators.py
│   ├── serialization.py
│   └── model_serialization.py
│
├── reliability/
│   ├── async_retry.py
│   └── backoff_jitter.py
│
└── projs/
    └── patient_api/
        ├── main.py
        ├── schemas.py
        └── patients.json
````

---

# Topics

## 1. Pydantic

Location:

```text
pydantic/
```

Hands-on implementations covering Pydantic models, nested models,
validation, and serialization.

### `pydantic_models.py`

Basic Pydantic model creation and type validation.

Path:

```text
pydantic/pydantic_models.py
```

### `nested_models.py`

Nested Pydantic models, recursive models, and forward references.

Path:

```text
pydantic/nested_models.py
```

### `pydantic_validators.py`

Custom validation using Pydantic validators.

Topics include:

* `field_validator`
* `model_validator`
* computed fields

Path:

```text
pydantic/pydantic_validators.py
```

### `serialization.py`

Pydantic model serialization and conversion.

Path:

```text
pydantic/serialization.py
```

### `model_serialization.py`

Additional experimentation with model serialization and representation.

Path:

```text
pydantic/model_serialization.py
```

---

# 2. FastAPI

Location:

```text
fastapi/
```

FastAPI fundamentals and asynchronous API development.

### `async_fastapi.py`

FastAPI endpoints using asynchronous Python patterns.

Path:

```text
fastapi/async_fastapi.py
```

### `dependency_injection.py`

FastAPI dependency injection patterns.

Path:

```text
fastapi/dependency_injection.py
```

---

# 3. Async Patterns

Location:

```text
async_patterns/
```

Core Python asynchronous programming and concurrency patterns.

### `semaphore_timeout.py`

Demonstrates:

* `asyncio.Semaphore`
* `asyncio.timeout()`
* `asyncio.gather()`
* latency measurement with `perf_counter()`
* concurrent async tasks

Path:

```text
async_patterns/semaphore_timeout.py
```

---

# 4. Reliability

Location:

```text
reliability/
```

Patterns for making asynchronous backend operations more reliable.

### `async_retry.py`

Basic asynchronous retry implementation.

Topics include:

* retry attempts
* maximum retry attempts
* retryable exceptions
* non-retryable exceptions
* `await`
* asynchronous retry delays

Path:

```text
reliability/async_retry.py
```

### `backoff_jitter.py`

Advanced retry strategy using:

* exponential backoff
* jitter
* concurrent async retries

Backoff pattern:

```text
1s → 2s → 4s → 8s
```

Path:

```text
reliability/backoff_jitter.py
```

---

# 5. Projects

Location:

```text
projs/
```

Small projects that combine multiple backend concepts.

## Patient API

Path:

```text
projs/patient_api/
```

Contains:

### `main.py`

Main FastAPI application and API endpoints.

### `schemas.py`

Pydantic schemas used by the patient API.

### `patients.json`

Local JSON data used by the application.

---