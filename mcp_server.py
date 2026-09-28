import sys
import json
from client import PPOClippedSurrogate

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-ppo-clipped-surrogate-objective-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "evaluate_ppo_clipping",
                        "description": "Compute PPO clipped surrogate objective and clipping metrics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "old_probs": {"type": "array", "items": {"type": "number"}},
                                "new_probs": {"type": "array", "items": {"type": "number"}},
                                "advantages": {"type": "array", "items": {"type": "number"}},
                                "clip_epsilon": {"type": "number", "default": 0.2}
                            },
                            "required": ["old_probs", "new_probs", "advantages"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "evaluate_ppo_clipping":
            old_p = args.get("old_probs", [])
            new_p = args.get("new_probs", [])
            adv = args.get("advantages", [])
            eps = args.get("clip_epsilon", 0.2)
            ppo = PPOClippedSurrogate(clip_epsilon=eps)
            res = ppo.compute_surrogate_loss(old_p, new_p, adv)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res)}]
                }
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
