import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig, ax = plt.subplots(1, 1)

fraction = 1  # .05

norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbar = ax.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap='gray'),
            ax=ax, pad=.05)

ticklabels = ['0', '0.5', '1.0']
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='gray'), cax=ax, orientation='vertical')
tick_locator = ticker.LinearLocator(numticks=3)
cb.locator = tick_locator
cb.update_ticks()
cb.set_ticklabels(ticklabels)

ax.axis('off')
plt.show()