import random
import scipy.stats as stats
import numpy as np

class Passenger:
    """Passenger class"""

    def __init__(self, id, origin, destination, request_time, price, entered=False, assign_time=None, wait_time=0, choice=None, max_wait=2) -> None:
        """
        price: price set for the trip
        choice: choice model for passenger
        max_wait: maximum waiting time
        """
        self.id = id
        self.origin = origin
        self.destination = destination
        self.request_time = request_time
        self.price = price
        self.entered = entered
        self.assign_time = assign_time
        self.wait_time = wait_time
        self.choice = choice
        self.max_wait = max_wait

    def unmatched_update(self):
        """Update state of passenger if not matched. Return True if maximum waiting time is reached otherwise False."""

        self.wait_time += 1
        if self.wait_time >= self.max_wait:
            return True
        else:
            return False

    def match(self, timestamp):
        """Update state of passenger once get matched. Return True if the passenger accept the price otherwise False."""
        accept = choice_passenger_accept(self.price, self.choice)
        if accept:
            self.assign_time = timestamp
            return True
        else:
            return False
        
    def enter(self, price=None):
        if price is not None:
            self.price = price
        enterq = choice_passenger_enter(self.price, self.choice)
        self.entered = enterq
        return enterq     

def choice_passenger_enter(price, mtype=None):
    """Choice model for passenger entering queue. Return True if enter else return False."""
    if mtype is None:
        # Use default exponential disteibution
        # reject_prob = stats.expon.cdf(price, scale=1/2)
        reject_prob = 0
        sample = random.uniform(0,1)
        if sample < reject_prob:
            return False
        else:
            return True
        
def choice_passenger_accept(price, mtype=None):
    """Choice model for passenger accepting ride match. Return True if accept else return False."""
    if mtype is None:
        # Use default exponential disteibution
        # reject_prob = stats.expon.cdf(price, scale=1/2)
        reject_prob = 0
        sample = random.uniform(0,1)
        if sample < reject_prob:
            return False
        else:
            return True



def generate_passenger(demand, max_wait=2, arrivals=None):
    """
    Generate passenger according to the demand

    demand: (origin,destination,time,total demand,price)
    arrivals: number of passengers already arrive in the system

    return: list of new passengers, total number of passenger arrivals
    """
    newp = []
    ori, des, t, d, p = demand
    for i in range(d):
        if arrivals is None:
            newp.append(Passenger(i, ori, des, t, p, max_wait=max_wait))
        else:
            newp.append(Passenger(arrivals+1, ori, des, t, p, max_wait=max_wait))
            arrivals += 1

    return newp, arrivals