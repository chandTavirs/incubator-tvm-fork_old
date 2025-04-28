import paramiko
import time
import socket


class RemoteRebooter:
    def __init__(self, host, username, password=None, key_file=None, port=22):
        """
        Initialize the SSH connection details.

        :param host: Remote server IP or hostname.
        :param username: SSH username.
        :param password: SSH password (optional if using key-based authentication).
        :param key_file: Path to the SSH private key file (optional).
        :param port: SSH port (default is 22).
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_file = key_file
        self.client = None

    def connect(self):
        """Establish an SSH connection to the remote server."""
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if self.key_file:
                self.client.connect(hostname=self.host, port=self.port, username=self.username,
                                    key_filename=self.key_file)
            else:
                self.client.connect(hostname=self.host, port=self.port, username=self.username, password=self.password)

            print("Connected to the remote server.")
            return True
        except Exception as e:
            print(f"SSH connection failed: {e}")
            return False

    def disconnect(self):
        """Close the SSH connection."""
        if self.client:
            self.client.close()
            print("Disconnected from the remote server.")

    def execute_command(self, command):
        """Execute a command on the remote server."""
        if self.client:
            stdin, stdout, stderr = self.client.exec_command(command)
            output = stdout.read().decode()
            error = stderr.read().decode()

            if output:
                print(f"OUTPUT: {output}")
            if error:
                print(f"ERROR: {error}")

    def reboot_and_run_script(self, script_command):
        """
        Reboots the server and runs a script after it comes back online.

        :param script_command: Command to run after reboot (e.g., "nohup sudo ./myscript.sh > myscript.log 2>&1 &").
        """
        reboot_command = "sudo reboot"

        if not self.connect():
            return

        print("Rebooting the server...")
        self.execute_command(reboot_command)
        self.disconnect()

        # Wait for the reboot to complete
        if self.wait_for_reboot():
            if self.connect():
                print("Executing the script after reboot...")
                self.execute_command(script_command)
                self.disconnect()
            else:
                print("Failed to reconnect after reboot.")
        else:
            print("Server did not come back online in time.")

    def wait_for_reboot(self, timeout=300):
        """Wait for the remote server to go down and come back up."""
        print("Waiting for the server to go down...")
        while True:
            try:
                with socket.create_connection((self.host, self.port), timeout=5):
                    pass
            except (socket.error, ConnectionRefusedError):
                break
            time.sleep(2)

        print("Server is down. Waiting for it to come back up...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((self.host, self.port), timeout=5):
                    print("Server is back online!")
                    return True
            except (socket.error, ConnectionRefusedError):
                time.sleep(5)

        print("Timeout: Server did not come back online in time.")
        return False