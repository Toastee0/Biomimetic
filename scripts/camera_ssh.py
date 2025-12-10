#!/usr/bin/env python3
"""SSH helper for camera access"""

import paramiko
import sys
import os

def ssh_execute(host, username, password, command):
    """Execute command via SSH and return output"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(host, username=username, password=password, timeout=10)
        stdin, stdout, stderr = client.exec_command(command)

        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        exit_code = stdout.channel.recv_exit_status()

        client.close()

        if error:
            print(error, file=sys.stderr)
        if output:
            print(output)

        return exit_code, output, error
    except Exception as e:
        print(f"SSH Error: {e}", file=sys.stderr)
        return 1, "", str(e)

if __name__ == "__main__":
    host = "192.168.2.140"
    username = "recamera"
    password = "ClaudeRocks!"

    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
    else:
        command = "uname -a && pwd && ls -la"

    exit_code, output, error = ssh_execute(host, username, password, command)
    sys.exit(exit_code)
