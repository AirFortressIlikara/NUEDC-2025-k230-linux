"""
Author: ilikara 3435193369@qq.com
Date: 2025-07-15 16:51:59
LastEditors: ilikara 3435193369@qq.com
LastEditTime: 2025-07-15 16:52:06
FilePath: /nuedc2025-linux/communicate.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import argparse
import logging
import struct
import sys
import termios
import fcntl
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SerialPort:
    def __init__(self, serial_port="/dev/ttyS3", baudrate=115200):
        """初始化串口
        :param port: 串口设备文件，如/dev/ttyS3
        :param baudrate: 波特率
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.fd = None

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

    def close(self):
        """关闭串口"""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        self.close()

    def send_command_with_3_floats(self, command_code, float1, float2, float3):
        """发送带有3个小端序浮点数的指令
        格式: 55 [1字节指令号] [3*小端序浮点数] 01
        """
        # 打包数据
        header = bytes([0x55, command_code])
        floats = struct.pack("<fff", float1, float2, float3)
        footer = bytes([0x01])

        # 组合完整数据包
        packet = header + floats + footer

        # 发送数据
        if self.ser_fd is not None:
            os.write(self.ser_fd, packet)
            logger.debug(f"Sent: {packet.hex(' ')}")

    def send_command_with_1_float(self, command_code, float1):
        """发送带有1个小端序浮点数的指令
        格式: 55 [1字节指令号] [1*小端序浮点数] 01
        """
        # 打包数据
        header = bytes([0x55, command_code])
        floats = struct.pack("<f", float1)
        footer = bytes([0x01])

        # 组合完整数据包
        packet = header + floats + footer

        # 发送数据
        if self.ser_fd is not None:
            os.write(self.ser_fd, packet)
            logger.debug(f"Sent: {packet.hex(' ')}")


def parse_user_input(user_input):
    """解析用户输入
    格式: 1/3 指令号(十六进制) 浮点数1 [浮点数2 浮点数3]
    例如: 3 01 1.23 4.56 7.89
         1 02 3.14
    """
    parts = user_input.strip().split()
    if len(parts) < 3:
        raise ValueError("Invalid input format")

    try:
        cmd_type = int(parts[0])
        if cmd_type not in (1, 3):
            raise ValueError("First argument must be 1 or 3")

        command_code = int(parts[1], 16)

        floats = []
        for part in parts[2:]:
            try:
                floats.append(float(part))
            except ValueError:
                raise ValueError(f"Invalid float value: {part}")

        if (cmd_type == 1 and len(floats) != 1) or (cmd_type == 3 and len(floats) != 3):
            raise ValueError(f"Expected {cmd_type} float(s), got {len(floats)}")

        return cmd_type, command_code, floats
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")


def main():
    parser = argparse.ArgumentParser(description="Serial to UDP Bridge")
    parser.add_argument(
        "--serial", "-s", default="/dev/ttyS3", help="Serial port (default: /dev/ttyS3)"
    )
    parser.add_argument(
        "--baud", "-b", default=115200, type=int, help="Baud rate (default: 115200)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    serial = SerialPort(serial_port=args.serial, baudrate=args.baud)

    logger.info(f"Serial port {args.serial} opened at {args.baud} baud")
    logger.info("Enter commands in the format:")
    logger.info("  1 <command_hex> <float>  - Send 1 float")
    logger.info("  3 <command_hex> <float1> <float2> <float3>  - Send 3 floats")
    logger.info("Example: 3 01 1.23 4.56 7.89")
    logger.info("Type 'exit' to quit")
    try:
        while True:
            user_input = input("> ")
            if user_input.lower() in ("exit", "quit"):
                break

            try:
                cmd_type, command_code, floats = parse_user_input(user_input)

                if cmd_type == 1:
                    serial.send_command_with_1_float(command_code, floats[0])
                else:
                    serial.send_command_with_3_floats(
                        command_code, floats[0], floats[1], floats[2]
                    )
            except ValueError as e:
                logger.error(f"Error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
    finally:
        serial.close()
        logger.info("Serial port closed")


# 使用示例
if __name__ == "__main__":
    main()
