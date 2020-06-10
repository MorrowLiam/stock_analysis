# %% codecell
import cvxpy as cp
# %% markdown
# 2. Convert the annualized average returns and the covariance matrix to `numpy` arrays:
# %% codecell
avg_returns = avg_returns.values
cov_mat = cov_mat.values
# %% markdown
# 3. Set up the optimization problem:
# %% codecell
weights = cp.Variable(n_assets)
gamma = cp.Parameter(nonneg=True)
portf_rtn_cvx = avg_returns * weights
portf_vol_cvx = cp.quad_form(weights, cov_mat)
objective_function = cp.Maximize(portf_rtn_cvx - gamma * portf_vol_cvx)
problem = cp.Problem(objective_function,
                     [cp.sum(weights) == 1, weights >= 0])
# %% markdown
# 4. Calculate the Efficient Frontier:
# %% codecell
N_POINTS = 25
portf_rtn_cvx_ef = np.zeros(N_POINTS)
portf_vol_cvx_ef = np.zeros(N_POINTS)
weights_ef = []
gamma_range = np.logspace(-3, 3, num=N_POINTS)

for i in range(N_POINTS):
    gamma.value = gamma_range[i]
    problem.solve()
    portf_vol_cvx_ef[i] = cp.sqrt(portf_vol_cvx).value
    portf_rtn_cvx_ef[i] = portf_rtn_cvx.value
    weights_ef.append(weights.value)
# %% markdown
# 5. Plot the allocation for different values of the risk-aversion parameter:
# %% codecell
weights_df = pd.DataFrame(weights_ef,
                          columns=RISKY_ASSETS,
                          index=np.round(gamma_range, 3))
ax = weights_df.plot(kind='bar', stacked=True)
ax.set(title='Weights allocation per risk-aversion level',
       xlabel=r'$\gamma$',
       ylabel='weight')
ax.legend(bbox_to_anchor=(1,1))

plt.tight_layout()
plt.savefig('images/ch7_im15.png')
plt.show()


# %% codecell
#MARKS = ['o', 'X', 'd', '*']

fig, ax = plt.subplots()
ax.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, 'g-')
for asset_index in range(n_assets):
     plt.scatter(x=np.sqrt(cov_mat[asset_index, asset_index]),
                 y=avg_returns[asset_index],
                 marker=MARKS[asset_index],
                 label=RISKY_ASSETS[asset_index],
                 s=150)
ax.set(title='Efficient Frontier',
       xlabel='Volatility',
       ylabel='Expected Returns', )
ax.legend()

plt.tight_layout()
plt.savefig('images/ch7_im16.png')
plt.show()
 

# # %% markdown
# # ### There's more
# # %% codecell
# x_lim = [0.25, 0.6]
# y_lim = [0.125, 0.325]
#
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(vols_range, rtns_range, 'g-', linewidth=3)
# ax[0].set(title='Efficient Frontier - Minimized Volatility',
#           xlabel='Volatility',
#           ylabel='Expected Returns',
#           xlim=x_lim,
#           ylim=y_lim)
#
# ax[1].plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, 'g-', linewidth=3)
# ax[1].set(title='Efficient Frontier - Maximized Risk-Adjusted Return',
#           xlabel='Volatility',
#           ylabel='Expected Returns',
#           xlim=x_lim,
#           ylim=y_lim)
#
# plt.tight_layout()
# plt.savefig('images/ch7_im17.png')
# plt.show()
# # %% codecell
# max_leverage = cp.Parameter()
# problem_with_leverage = cp.Problem(objective_function,
#                                    [cp.sum(weights) == 1,
#                                     cp.norm(weights, 1) <= max_leverage])
# # %% codecell
# LEVERAGE_RANGE = [1, 2, 5]
# len_leverage = len(LEVERAGE_RANGE)
# N_POINTS = 25
#
# portf_vol_l_ef = np.zeros((N_POINTS, len_leverage))
# portf_rtn_l_ef = np.zeros(( N_POINTS, len_leverage))
# weights_ef = np.zeros((len_leverage, N_POINTS, n_assets))
#
# for lev_ind, leverage in enumerate(LEVERAGE_RANGE):
#     for gamma_ind in range(N_POINTS):
#         max_leverage.value = leverage
#         gamma.value = gamma_range[gamma_ind]
#         problem_with_leverage.solve()
#         portf_vol_l_ef[gamma_ind, lev_ind] = cp.sqrt(portf_vol_cvx).value
#         portf_rtn_l_ef[gamma_ind, lev_ind] = portf_rtn_cvx.value
#         weights_ef[lev_ind, gamma_ind, :] = weights.value
#
# # %% codecell
# fig, ax = plt.subplots()
#
# for leverage_index, leverage in enumerate(LEVERAGE_RANGE):
#     plt.plot(portf_vol_l_ef[:, leverage_index],
#              portf_rtn_l_ef[:, leverage_index],
#              label=f'{leverage}')
#
# ax.set(title='Efficient Frontier for different max leverage',
#        xlabel='Volatility',
#        ylabel='Expected Returns')
# ax.legend(title='Max leverage')
#
# plt.tight_layout()
# plt.savefig('images/ch7_im18.png')
# plt.show()
# # %% codecell
# fig, ax = plt.subplots(len_leverage, 1, sharex=True)
#
# for ax_index in range(len_leverage):
#     weights_df = pd.DataFrame(weights_ef[ax_index],
#                               columns=RISKY_ASSETS,
#                               index=np.round(gamma_range, 3))
#     weights_df.plot(kind='bar',
#                     stacked=True,
#                     ax=ax[ax_index],
#                     legend=None)
#     ax[ax_index].set(ylabel=(f'max_leverage = {LEVERAGE_RANGE[ax_index]}'
#                              '\n weight'))
#
#
# ax[len_leverage - 1].set(xlabel=r'$\gamma$')
# ax[0].legend(bbox_to_anchor=(1,1))
# ax[0].set_title('Weights allocation per risk-aversion level',
#                 fontsize=16)
#
# plt.tight_layout()
# plt.savefig('images/ch7_im19.png')
# plt.show()
# # %% codecell
