import RPi.GPIO as GPIO
import time
import curses

GPIO.setmode(GPIO.BCM)

# ===== Pin Setup (YOUR ORDER) =====
ENA = 17
IN1 = 27
IN2 = 22
IN3 = 26
IN4 = 6
ENB = 5

GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# PWM Speed Control
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)

pwmA.start(80)
pwmB.start(80)

# ===== Movement Functions =====

def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# ===== Keyboard Control =====
def main(stdscr):

    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.addstr("Arrow keys to move | SPACE = Stop | Q = Quit\n")

    try:
        while True:
            key = stdscr.getch()

            if key == curses.KEY_UP:
                stdscr.addstr(2, 0, "Forward   ")
                forward()

            elif key == curses.KEY_DOWN:
                stdscr.addstr(2, 0, "Backward  ")
                backward()

            elif key == curses.KEY_LEFT:
                stdscr.addstr(2, 0, "Left      ")
                left()

            elif key == curses.KEY_RIGHT:
                stdscr.addstr(2, 0, "Right     ")
                right()

            elif key == ord(' '):
                stdscr.addstr(2, 0, "Stop      ")
                stop()

            elif key == ord('q'):
                break

            time.sleep(0.1)

    finally:
        stop()
        GPIO.cleanup()

curses.wrapper(main)
