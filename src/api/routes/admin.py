# Admin API Endpoints

## Health Check

### GET /health
Gets the health status of the system.

#### Response
- 200: Healthy
- 503: Unhealthy

## System Management

### POST /admin/manage
Perform system management actions.

#### Request Body
- action: String indicating the action to be taken.

#### Response
- 200: Action completed successfully.
- 400: Invalid action.

### GET /admin/stats
Get system statistics.

#### Response
- 200: Returns system statistics such as uptime, resource usage, etc.

## Sample Response Structure
```json
{
  "status": "healthy",
  "uptime": "1234 hours",
  "resourceUsage": {
    "cpu": "10%",
    "memory": "50MB"
  }
}
```
