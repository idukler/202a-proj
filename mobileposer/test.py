from flask import Flask, request

app = Flask(__name__)


@app.route("/data", methods=["POST"])
def receive_data():
    # Print raw JSON exactly as sent by client
    data = request.get_json(force=True, silent=True)
    print("\n================= NEW /data POST =================")
    print(data)
    print("==================================================\n")

    # Optionally print some helpful breakdown if payload exists
    if isinstance(data, dict) and "payload" in data:
        print("Payload length:", len(data["payload"]))
        for i, item in enumerate(data["payload"]):
            name = item.get("name")
            values = item.get("values")
            print(f"  [{i}] name={name}, values={values}")

    return {"status": "ok"}, 200


if __name__ == "__main__":
    # Run a simple HTTP server listening on all interfaces, port 8000
    # Point your phone/watch app to: http://<this_machine_ip>:8000/data
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
