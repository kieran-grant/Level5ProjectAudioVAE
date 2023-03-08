from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, CustomJS, TapTool
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.layouts import column
from bokeh.models.widgets import Audio

# Define the audio files
audio_files = ['./test/debussy.wav', './test/duke.wav', './test/redhot.wav']

# Define the x and y coordinates of the points
x = [1, 2, 3]
y = [4, 5, 6]

# Create a ColumnDataSource object with the x and y coordinates
source = ColumnDataSource(data=dict(x=x, y=y))

# Create the scatter plot
plot = figure(plot_width=400, plot_height=400, tools='tap')

# Add the points to the plot
plot.circle('x', 'y', source=source)

# Define the JavaScript code that will play the audio file
code = """
var audio = new Audio();
audio.src = '%s';
audio.play();
"""

# Create a CustomJS callback that will play the audio file when a point is clicked
callback = CustomJS(args=dict(source=source), code=code)

# Add the TapTool to the plot and attach the callback
plot.add_tools(TapTool(callback=callback))

# Create an Audio widget for each audio file
audios = [Audio(url=audio_file, autoplay=False) for audio_file in audio_files]

# Create a layout with the plot and the Audio widgets
layout = column(plot, *audios)

# Output the plot and the Audio widgets to a file
output_file("scatter_plot_with_audio.html", resources=INLINE)

# Show the plot and the Audio widgets
show(layout)
