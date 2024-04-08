import matplotlib
import matplotlib.pyplot as plt
import numpy as np

colors = ["darkcyan", "orange", "pink"]
colors_pred = ["purple", "firebrick", "olive"]
colors_u = ["brown", "green", "yellow"]

def latexify():
    params = {
        'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
        'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

def plot_data(us, ys,
              ys_true=None, dt=1., u_other_scale=None,
              figsize=(8, 4), ax=None,
              char="-", alpha_u=1.,
              title=None, xlabel="time (sec)",
              ylabel="", ulabels=None,
              legend=True, legend_order=None, ax4legend=None
    ):
    if ax is None:
        if ax4legend is not None and ax4legend == "new":
            fig, axs = plt.subplots(1,2, figsize=figsize,
                sharex=True, gridspec_kw={'width_ratios':[200, 1]}
            )
            ax = axs[0]
            ax4legend = axs[1]

            ax4legend.axis("off")
        else:
            fig, ax = plt.subplots(figsize=figsize)
        new_fig = True
        if title is not None:
            fig.suptitle(title)
    else:
        new_fig = False

    N, ny = ys.shape
    N, nu = us.shape
    ts = np.arange(N) * dt
    if us is not None:
        nu = us.shape[1]
        if ulabels is None:
            ulabels = [r"$u_{"+str(j+1)+r", k}$" for j in range(nu)]
        if u_other_scale is not None:
            axu = ax.twinx()
        for j in range(nu):
            ax_ = ax
            if u_other_scale is not None and j in u_other_scale:
                ax_ = axu
            ax_.plot(ts[:N], us[:, j], char, label=ulabels[j], alpha=alpha_u, color=colors_u[j])
    for i in range(ny):
        alphay = 1 #  / (i+1)
        ax.plot(ts, ys[:N, i], char, label=r"$y_{"+str(i+1)+r", k}$", color=colors[i], alpha=alphay)
        if ys_true is not None:
            ax.plot(ts, ys_true[:N, i], char, label=f"y unnoised {i+1}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    h, l = ax.get_legend_handles_labels()
    if u_other_scale is not None:
        h_v, l_v  = axu.get_legend_handles_labels()
        h = [ h_v[0] ] +  h
        l = [ l_v[0] ] +  l
    if legend_order is not None:
        h = [h[i] for i in legend_order]
        l = [l[i] for i in legend_order]

    if legend:
        if ax4legend is None:
            ax4legend = ax
            box = (0, 0.8)
        else:
            box = (0, 0.5)
        ax4legend.legend(h, l, loc='center left', bbox_to_anchor=box)

    if new_fig:
        fig.tight_layout()
        return fig
    return None

def plot_est(us, ys, yest,
              ys_true=None, pred=None, Spred=None,
              nstd=2, dt=1., u_other_scale=None,
              figsize=(8, 4), ax=None,
              char="-", alpha_u=1., alpha_pred=1.,
              title=None, xlabel="time (sec)",
              ylabel="", ulabels=None,
              legend=True, legend_order=None, ax4legend=None
    ):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        new_fig = True
        if title is not None:
            fig.suptitle(title)
    else:
        new_fig = False

    linewidth_pred = 1. # used to be 3
    N, ny = ys.shape
    N, nu = us.shape
    ts = np.arange(N)*dt
    if us is not None:
        nu = us.shape[1]
        if ulabels is None:
            ulabels = [r"$u_{"+str(j+1)+r", k}$" for j in range(nu)]
        if u_other_scale is not None:
            axu = ax.twinx()
        for j in range(nu):
            ax_ = ax
            if u_other_scale is not None and j in u_other_scale:
                ax_ = axu
            ax_.plot(ts, us[:, j], char, label=ulabels[j],
                     alpha=alpha_u, color=colors_u[j])

    if pred is not None:
        tspred, yspred = pred
        t0s, y0s = np.empty(len(tspred)), np.empty((len(tspred), ny))
        tpred_stack, ypred_stack = [], []
        for k, (tpred, ypred) in enumerate(zip(tspred, yspred)):
            y0s[k, :] =  ypred[0]
            t0s[k] = tpred[0]
            tpred_stack = tpred_stack + list(tpred) +  [np.nan]
            for y in ypred:
                ypred_stack.append(y.reshape(ny) )
            ypred_stack.append(np.ones(ny)*np.nan)
        tpred_stack = np.array(tpred_stack)
        ypred_stack = np.array(ypred_stack).reshape(-1, ny)
        if Spred is not None:
            spred_stack = []
            for k, spred in enumerate(Spred):
                for s in spred:
                    sdiag = np.sqrt(np.diag(s))
                    spred_stack.append(sdiag.reshape(ny) )
                spred_stack.append(np.ones(ny)*np.nan)
            spred_stack = np.array(spred_stack).reshape(-1, ny)

    for i in range(ny):
        if ys is not None:
            ax.plot(ts * dt, ys[:N, i], char, label=r"$y^{"+str(i+1)+r"}_k$", color=colors[i])
        if yest is not None:
            ax.plot(ts * dt, yest[:N, i], char, label=r"$\hat{y}^{"+str(i+1)+r"}$", color=colors[i])
        if ys_true is not None:
            ax.plot(ts * dt, ys_true[:N, j], char, label=f"y unnoised {j}")
        if pred is not None:
            ax.plot(t0s * dt, y0s[:, i], 'o', color=colors_pred[i], ) #label=r"$t$")
            ax.plot(tpred_stack * dt, ypred_stack[:, i], alpha=alpha_pred,
                label=r"$\hat{y}^{"+str(i+1)+r"}_{k\mid t}$", color=colors_pred[i],
                linewidth=linewidth_pred)
        if Spred is not None:
            alpha_tube = 0.5
            yhat = ypred_stack[:, i]
            yerr = nstd*spred_stack[:, i]
            ax.fill_between(
                tpred_stack * dt, yhat-yerr, yhat+yerr, alpha=alpha_tube, color=colors_pred[i],
                label=r"$\hat{y}^{" +str(i+1)+ r"}_{k\mid t} \pm" + str(nstd) + r"\sigma$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    h, l = ax.get_legend_handles_labels()
    if u_other_scale is not None:
        h_v, l_v  = axu.get_legend_handles_labels()
        h = [ h_v[0] ] +  h
        l = [ l_v[0] ] +  l
    if legend_order is None:
        h_, l_ = h, l
    else:
        h_ = [h[i] for i in legend_order]
        l_ = [l[i] for i in legend_order]

    if legend:
        if ax4legend is None:
            ax4legend = ax
            box = (1, 0.5)
        else:
            box = (1, 0.5)
        ax4legend.legend(h_, l_, loc='center left', bbox_to_anchor=box)

    if new_fig:
        fig.tight_layout()
        return fig
    return None
