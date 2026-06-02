# ─────────────────────────────────────────────────────────────────────────────
#  Traffic Light Controller — MicroPython (ESP32-S3)
#
#  Wiring (from your table):
#    GPIO  4  →  220Ω  →  Green  LED(+)  →  GND
#    GPIO  5  →  220Ω  →  Yellow LED(+)  →  GND
#    GPIO  2  →  220Ω  →  Red    LED(+)  →  GND
#    GPIO 15  →  Pushbutton  →  GND          (active LOW, PULL_UP used)
#
#  Normal cycle  : GREEN (5 s) → YELLOW (2 s) → RED (5 s) → repeat
#  Emergency     : Button press → print EMERGENCY → GREEN (5 s) → resume cycle
#  Startup       : All LEDs OFF
#  Serial output : STARTUP / GREEN / YELLOW / RED / EMERGENCY
# ─────────────────────────────────────────────────────────────────────────────

from machine import Pin
import time

# ── Pin Definitions ───────────────────────────────────────────────────────────
GREEN_PIN  = 4
YELLOW_PIN = 5
RED_PIN    = 2
BTN_PIN    = 15

# ── Hardware Init ─────────────────────────────────────────────────────────────
green  = Pin(GREEN_PIN,  Pin.OUT)
yellow = Pin(YELLOW_PIN, Pin.OUT)
red    = Pin(RED_PIN,    Pin.OUT)

# Button wired: GPIO 15 → Button → GND
# PULL_UP keeps pin HIGH at rest; pressing pulls it LOW
button = Pin(BTN_PIN, Pin.IN, Pin.PULL_UP)

# ── Timing Constants (milliseconds) ──────────────────────────────────────────
T_GREEN  = 5000
T_YELLOW = 2000
T_RED    = 5000

T_DEBOUNCE = 30   # ms — ignore glitches shorter than this

# ── LED Helpers ───────────────────────────────────────────────────────────────
def all_off():
    """Turn every LED off."""
    green.value(0)
    yellow.value(0)
    red.value(0)

def set_state(label):
    """
    Activate the correct LED and print state to serial.
    'label' must be one of: "GREEN" | "YELLOW" | "RED"
    """
    all_off()
    if   label == "GREEN":  green.value(1)
    elif label == "YELLOW": yellow.value(1)
    elif label == "RED":    red.value(1)
    print(label)

# ── Button Helper ─────────────────────────────────────────────────────────────
def btn_down():
    """
    Returns True if button is pressed (active LOW + debounced).
    Non-blocking — call repeatedly from a polling loop.
    """
    if button.value() == 0:            # first read: pin pulled LOW
        time.sleep_ms(T_DEBOUNCE)
        return button.value() == 0     # confirm after debounce window
    return False

# ── Phase Runner ──────────────────────────────────────────────────────────────
def run_phase(label, duration_ms):
    """
    Activate `label` LED state and hold for `duration_ms` milliseconds,
    polling the button every 10 ms throughout.

    Returns:
        False  — phase completed normally (full duration elapsed)
        True   — button was pressed mid-phase (waits for button release
                 before returning, so no double-trigger)
    """
    set_state(label)

    deadline = time.ticks_add(time.ticks_ms(), duration_ms)

    while time.ticks_diff(deadline, time.ticks_ms()) > 0:
        if btn_down():
            # Wait for the button to be fully released before continuing
            while button.value() == 0:
                time.sleep_ms(10)
            return True          # signal: interrupted

        time.sleep_ms(10)        # polling interval

    return False                 # signal: completed normally

# ── Emergency Handler ─────────────────────────────────────────────────────────
def do_emergency():
    """
    Called whenever the button fires during any traffic phase.

    Behaviour:
      1. Print "EMERGENCY" to serial
      2. Immediately switch to GREEN
      3. Hold GREEN for T_GREEN ms (button ignored during this window
         to prevent re-triggering within the same override)
    """
    print("EMERGENCY")
    set_state("GREEN")
    time.sleep_ms(T_GREEN)      # blocking — intentional; no re-entry here

# ── Startup ───────────────────────────────────────────────────────────────────
all_off()
print("STARTUP")

# ── Main Loop ─────────────────────────────────────────────────────────────────
# After any emergency the cycle restarts from GREEN.
# Serial will show GREEN twice (once inside do_emergency, once at cycle top)
# — that correctly reflects the state each time it changes.

while True:

    # ── GREEN  (5 s) ──────────────────────────────────────────────────────────
    if run_phase("GREEN", T_GREEN):
        do_emergency()
        continue            # restart: GREEN → YELLOW → RED → …

    # ── YELLOW (2 s) ──────────────────────────────────────────────────────────
    if run_phase("YELLOW", T_YELLOW):
        do_emergency()
        continue

    # ── RED    (5 s) ──────────────────────────────────────────────────────────
    if run_phase("RED", T_RED):
        do_emergency()
        continue
