import tensorflow as tf

from logger.SummaryType import SummaryType


class LoggerNew:

    def __init__(self, session, writer):
        self.session = session
        self.writer = writer
        self.summary_dict = {}

    def write_to_log(self, input, name, iteration, type=SummaryType.SCALAR):
        """
        Writes the given input to the log with name and iteration specified
        Takes care of the tensorboard summary / placeholder management.

        :param input: Value that should be saved to the log
        :param name: What name the value should have
        :param iteration: Current iteration
        :param type: type of the summary data
        """
        if self.summary_dict.get(name) is None:
            self._add_summary(input, name, type)
        self._run_summary(input,name,iteration)

    def _add_summary(self, input, name, type):
        """
        Adds a summary and placeholder to the logger for the given input and name

        :param type: Type of summary to add
        :param input: Values that should be saved to the log
        :param name: Name for the value
        """
        if type == SummaryType.TEXT:
            summary = tf.summary.text(name,tf.convert_to_tensor(input))
            placeholder = tf.placeholder(name=name + "placeholder", dtype=tf.string)
        elif type == SummaryType.SCALAR:
            placeholder = tf.placeholder(name=name + "placeholder", dtype=tf.float32)
            summary = tf.summary.scalar(name,placeholder)
        elif type == SummaryType.HISTOGRAM:
            placeholder = tf.placeholder(name=name + "placeholder", dtype=tf.float32)
            summary = tf.summary.histogram(name, placeholder)
        elif type == SummaryType.IMAGE:
            placeholder = tf.placeholder(name=name + "placeholder", dtype=tf.float32)
            summary = tf.summary.image(name, placeholder)
        elif type == SummaryType.NON_TENSOR:
            placeholder = tf.placeholder(name=name+"placeholder",dtype=tf.float32)
            summary = tf.summary.scalar(name, placeholder)
        else:
            raise ValueError("No summary for input with type :"+repr(type))

        self.summary_dict[name] = (summary,placeholder)

    def _run_summary(self, input, name, iteration):
        """
        Runs a saved summary based on the name with the input data and iteration counter.

        :param input: values to be feeded to the log
        :param name: the name of the summary
        :param iteration: counter for the iteration
        """
        (summary,placeholder) = self.summary_dict[name]
        result = self.session.run(summary,feed_dict={placeholder:input})
        self.writer.add_summary(result,iteration)
