import numpy as np
import matplotlib.pyplot as plt

# T-Test

# Mittelwert x berechnen
def mean(values):
  return 1/len(values) * sum(values)

# Standardabweichung s berechnen
def std(values):
  m = mean(values)
  return (1/(len(values)-1) * sum((x - m)**2 for x in values))**0.5

# t berechnen
def t(x, u, s, N):
  return (x - u)/s * N**0.5

# plotten

def simple_plot(values, mean, std_def):
  x = np.linspace(mean - 4*std_def, mean + 4*std_def, 100)

  # plot normal distribution
  plt.plot(x, norm.pdf(x, mean, std_def))

  x_values = [-3.45, 3.45]  # replace with your x values
  # add axis
  for value in x_values:
    plt.axvline(x=value, color='r', linestyle='--')

  # add title and labels
  plt.title('Wahrscheinlichkeit für Abweichung < 3.45mm')
  plt.xlabel('x')
  plt.ylabel('Abweichung in mm')

  # fill area
  x_fill = np.linspace(-3.45, 3.45, 100)
  y_fill = norm.pdf(x_fill, mean, std_def)
  plt.fill_between(x_fill, y_fill, color='lightblue')

  # add grid
  plt.grid(True)

  # show plot
  plt.show()

#  QQ-Plot
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def simple_qq_plot_example(data):

  # Erstellen des QQ-Plots
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  stats.probplot(data, dist="norm", plot=ax)
  ax.get_lines()[1].set_color('red')  # Linie für die erwartete Normalverteilung hinzufügen (optional)
  plt.title('QQ-Plot')
  plt.xlabel('Theoretische Quantile')
  plt.ylabel('Beobachtete Quantile')
  plt.grid(True)
  plt.show()

# standard deviation
# std = np.std(values)