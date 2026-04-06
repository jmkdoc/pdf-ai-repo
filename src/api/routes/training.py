# Training API

This module contains endpoints for managing training sessions on PDF collections.

## Endpoints

### 1. Create Training Session
- **URL:** `/api/training/create`
- **Method:** `POST`
- **Description:** Creates a new training session for a PDF collection.
- **Request Body:**
    ```json
    {
        "collectionId": "string",
        "sessionName": "string",
        "parameters": {
            // training parameters
        }
    }
    ```
- **Response:** 201 Created
    ```json
    {
        "sessionId": "string",
        "status": "success",
        "message": "Training session created."
    }
    ```

### 2. Get Training Session Status
- **URL:** `/api/training/status/{sessionId}`
- **Method:** `GET`
- **Description:** Retrieves the status of a specific training session.
- **Response:** 200 OK
    ```json
    {
        "sessionId": "string",
        "status": "string",
        "logs": [ "string" ]
    }
    ```

### 3. List All Training Sessions
- **URL:** `/api/training/sessions`
- **Method:** `GET`
- **Description:** Lists all training sessions.
- **Response:** 200 OK
    ```json
    [
        {
            "sessionId": "string",
            "sessionName": "string",
            "status": "string"
        }
    ]
    ```

### 4. Delete Training Session
- **URL:** `/api/training/delete/{sessionId}`
- **Method:** `DELETE`
- **Description:** Deletes a specific training session.
- **Response:** 200 OK
    ```json
    {
        "status": "success",
        "message": "Training session deleted."
    }
    ```
