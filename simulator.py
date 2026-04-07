import statistics
import random

class TrafficSimulator:
    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self.lanes = ["N", "S", "E", "W"]
        self.queues = {lane: 0 for lane in self.lanes}
        self.wait_times = {lane: 0.0 for lane in self.lanes}
        self.signal_phase = "NS_GREEN"
        self.time_since_last_change = 0
        self.emergency = {lane: False for lane in self.lanes}
        self.pedestrians = {lane: False for lane in self.lanes}
        self.step_count = 0
        
        if difficulty == "easy":
            self.arrival_rates = {"N": 2, "S": 2, "E": 0, "W": 0}
        elif difficulty == "medium":
            self.arrival_rates = {"N": 2, "S": 2, "E": 2, "W": 2}
        else: # hard
            self.arrival_rates = {"N": 3, "S": 3, "E": 3, "W": 3}
            
    def step(self, action_type: str) -> float:
        """Executes a simulation step and returns the reward."""
        phase_switches = 0
        
        if action_type == "SWITCH_PHASE":
            self.signal_phase = "EW_GREEN" if self.signal_phase == "NS_GREEN" else "NS_GREEN"
            self.time_since_last_change = 0
            phase_switches = 1
        else:
            self.time_since_last_change += 1
            
        self.step_count += 1
        
        # Deterministic arrivals
        arrivals = self._get_arrivals()
        for lane, count in arrivals.items():
            self.queues[lane] += count
            
        # Hard mode logic: emergencies and pedestrians
        if self.difficulty == "hard":
            # Using step_count to ensure purely deterministic behavior
            if self.step_count % 5 == 0:
                self.emergency["N"] = True
            if self.step_count % 7 == 0:
                self.pedestrians["E"] = True
            if self.step_count % 11 == 0:
                self.emergency["E"] = True
            if self.step_count % 3 == 0:
                self.pedestrians["N"] = True
                
        passed = {lane: 0 for lane in self.lanes}
        green_lanes = ["N", "S"] if self.signal_phase == "NS_GREEN" else ["E", "W"]
        red_lanes = ["E", "W"] if self.signal_phase == "NS_GREEN" else ["N", "S"]
        
        emergency_cleared = 0
        pedestrians_served = 0
        
        # Process green lanes (cars move)
        for lane in green_lanes:
            if self.pedestrians[lane]:
                pass # Cars must wait for pedestrians
            else:
                cars_can_pass = 3
                passed_cars = min(self.queues[lane], cars_can_pass)
                self.queues[lane] -= passed_cars
                passed[lane] = passed_cars
                
                if self.emergency[lane]:
                    emergency_cleared += 1
                    self.emergency[lane] = False
                    
        # Pedestrians waiting on red lanes can cross safely
        for lane in red_lanes:
             if self.pedestrians[lane]:
                 pedestrians_served += 1
                 self.pedestrians[lane] = False

        # Accumulate wait times based on queue (proxy for Little's law)
        for lane in self.lanes:
            self.wait_times[lane] += self.queues[lane]

        # Reward Calculation
        # Weights from problem statement
        alpha = 1.0
        beta = 0.5
        gamma = 0.2
        delta = 10.0
        eta = 3.0
        lam = 0.1
        mu = 0.05
        
        N_passed = sum(passed.values())
        sum_q = sum(self.queues.values())
        sum_w = sum(self.wait_times.values())
        N_emergency = emergency_cleared
        N_ped = pedestrians_served
        
        wait_values = list(self.wait_times.values())
        var_wait = statistics.variance(wait_values) if len(wait_values) > 1 else 0.0
        
        reward = (alpha * N_passed 
                  - beta * sum_q 
                  - gamma * sum_w 
                  + delta * N_emergency 
                  + eta * N_ped 
                  - lam * var_wait 
                  - mu * phase_switches)
                  
        return reward
        
    def _get_arrivals(self):
        """Deterministic arrivals."""
        arrivals = {}
        for lane, base_rate in self.arrival_rates.items():
            if base_rate > 0:
                # Add simple cyclical variation to arrivals
                offset = self.step_count % 3
                arrivals[lane] = max(0, base_rate - 1 + offset)
            else:
                arrivals[lane] = 0
        return arrivals
