from app.blockchain import verify_chain
from app.device_api import call_function


def send_security_alert(
    home_id,
    device_id,
    alert_type,
    severity,
    message,
    details=None,
):
    call_function("deviceSendSecurityAlert", {
        "alertType": alert_type,
        "severity": severity,
        "message": message,
        "details": details or {},
    })

    print(f"Security alert sent via API: {message}")


def check_tampering_and_alert(home_id, device_id):
    print("=== Checking blockchain/log integrity ===")

    # Important:
    # Verify only logs for the current home_id + device_id.
    result = verify_chain(home_id, device_id)

    tampered = False
    message = None

    if not result.get("valid", True):
        tampered = True
        message = result.get(
            "reason",
            "Local access log chain tampering detected.",
        )

    elif result.get("firebase_anchor_valid") is False:
        tampered = True
        message = result.get(
            "reason",
            "Firebase blockchain anchor mismatch detected.",
        )

    if tampered:
        print(" TAMPERING DETECTED")
        print(message)

        send_security_alert(
            home_id=home_id,
            device_id=device_id,
            alert_type="tampering_detected",
            severity="critical",
            message=message,
            details=result,
        )
    else:
        print("Blockchain integrity OK")

    return result
