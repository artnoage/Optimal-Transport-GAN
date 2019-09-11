import time
import datetime
class TimeClock:

    time_dict = {}


    def t(self,name):
        """
        Prints the current time of a timer with the given name or sets a timer with the name specified.
        Mein purpose of is to easily debug the run time of parts of the code.

        :param name: the name of the timer
        :return: no return
        """
        entry = self.time_dict.pop(name,None)
        if entry is None:
            self.time_dict.update({name:datetime.datetime.now()})
        else:
            elapsed = datetime.datetime.now() - entry
            print(str(name)+" ",elapsed.seconds, ":", elapsed.microseconds)

