'''
Author: ilikara 3435193369@qq.com
Date: 2025-07-15 16:45:50
LastEditors: ilikara 3435193369@qq.com
LastEditTime: 2025-07-15 16:46:27
FilePath: /nuedc2025-linux/bridge.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python3
import os
import select
import socket
import threading
import argparse
import logging
import fcntl
import termios
from time import sleep

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SerialUDPBridge:
    def __init__(self, serial_port, baudrate, udp_host, udp_port):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.remote_host = None
        self.remote_port = None

        # 初始化串口（使用文件接口）
        try:
            self.ser_fd = os.open(self.serial_port, os.O_RDWR | os.O_NOCTTY)

            # 配置串口参数
            attrs = termios.tcgetattr(self.ser_fd)

            # 设置波特率
            from termios import B9600, B19200, B38400, B57600, B115200

            baud_map = {
                9600: B9600,
                19200: B19200,
                38400: B38400,
                57600: B57600,
                115200: B115200,
            }
            if baudrate in baud_map:
                baud_const = baud_map[baudrate]
                attrs[4] = baud_const  # 输入波特率
                attrs[5] = baud_const  # 输出波特率

            # 8N1配置
            attrs[0] &= ~(
                termios.IGNBRK
                | termios.BRKINT
                | termios.PARMRK
                | termios.ISTRIP
                | termios.INLCR
                | termios.IGNCR
                | termios.ICRNL
                | termios.IXON
            )
            attrs[1] &= ~termios.OPOST
            attrs[2] &= ~(termios.CSIZE | termios.PARENB)
            attrs[2] |= termios.CS8
            attrs[3] &= ~(
                termios.ECHO
                | termios.ECHONL
                | termios.ICANON
                | termios.ISIG
                | termios.IEXTEN
            )

            termios.tcsetattr(self.ser_fd, termios.TCSANOW, attrs)

            # 设置非阻塞
            flags = fcntl.fcntl(self.ser_fd, fcntl.F_GETFL)
            fcntl.fcntl(self.ser_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        except Exception as e:
            logger.error(f"Failed to open serial port: {e}")
            raise

        # 初始化UDP socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(("0.0.0.0", self.udp_port))

        # 退出标志
        self.exit_flag = False

    def serial_to_udp(self):
        """从串口读取数据并发送到UDP"""
        logger.info(
            f"Serial to UDP thread started (Serial: {self.serial_port} -> UDP: {self.udp_host}:{self.udp_port})"
        )
        while not self.exit_flag:
            try:
                # 使用select检查是否有数据可读
                rlist, _, _ = select.select([self.ser_fd], [], [], 1.0)
                if self.ser_fd in rlist:
                    data = os.read(self.ser_fd, 4096)
                    if data:
                        if self.remote_host != None:
                            self.udp_socket.sendto(
                                data, (self.remote_host, self.remote_port)
                            )
                            logger.debug(f"Serial->UDP: Sent {len(data)} bytes")
            except Exception as e:
                logger.error(f"Error in serial_to_udp: {e}")
                sleep(1)

    def udp_to_serial(self):
        """从UDP读取数据并发送到串口"""
        logger.info(
            f"UDP to Serial thread started (UDP: {self.udp_port} -> Serial: {self.serial_port})"
        )
        while not self.exit_flag:
            try:
                data, addr = self.udp_socket.recvfrom(4096)
                if data:
                    os.write(self.ser_fd, data)
                    logger.debug(f"UDP->Serial: Received {len(data)} bytes from {addr}")
                    self.remote_host, self.remote_port = addr
            except Exception as e:
                logger.error(f"Error in udp_to_serial: {e}")
                sleep(1)

    def start(self):
        """启动转发服务"""
        logger.info("Starting Serial-UDP bridge...")

        # 创建并启动线程
        serial_thread = threading.Thread(target=self.serial_to_udp, daemon=True)
        udp_thread = threading.Thread(target=self.udp_to_serial, daemon=True)

        serial_thread.start()
        udp_thread.start()

        try:
            # 主线程等待
            while True:
                sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.exit_flag = True
            serial_thread.join()
            udp_thread.join()

            # 关闭资源
            os.close(self.ser_fd)
            self.udp_socket.close()
            logger.info("Bridge stopped cleanly")


def main():
    parser = argparse.ArgumentParser(description="Serial to UDP Bridge")
    parser.add_argument(
        "--serial", "-s", default="/dev/ttyS3", help="Serial port (default: /dev/ttyS3)"
    )
    parser.add_argument(
        "--baud", "-b", default=115200, type=int, help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--udp-host",   
        "-uh",
        default="127.0.0.1",
        help="UDP target host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--udp-port",
        "-up",
        default=8888,
        type=int,
        help="UDP listening port (default: 8888)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    bridge = SerialUDPBridge(
        serial_port=args.serial,
        baudrate=args.baud,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
    )

    bridge.start()


if __name__ == "__main__":
    main()
