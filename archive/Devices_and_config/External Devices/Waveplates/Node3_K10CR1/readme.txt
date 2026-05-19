
0. Install all necessary python packages: 
    from sipyco.pc_rpc import simple_server_loop
    from pylablib.devices import Thorlabs 

1. Add device_db["k10cr1_ndsp"] to your device_db, launcher_multi_rotor.py, k10cr1_drive.py, and k10cr1_functions.py to your repo
2. In your virtual environment where ARTIQ is installed, cd to the directory shown.
    - open a conda prompt
    - cd C:\Networking Experiment\artiq codes\artiq-
    master\repository\qn_artiq_routines\utilities\ndsp\thorlabs\with_kasli\two_k10cr1_one_server
    - conda activate artiq7
    - python launcher_multi_rotor.py
3. In a command prompt, run the NDSP server with python launcher_multi_rotor.py
4. Check Parameters match Eunji's ppt, if not use APT software to adjust settings to match.
5. In your experimental code:
    - Top of code:
    from k10cr1_functions import *
    - In def build(self):
            try: 
                self.setattr_device("k10cr1_ndsp")
            except Exception as e:
                print(f"Error connecting to device {e}")


