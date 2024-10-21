import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science', 'no-latex'])

default_batch_size = 2048
default_iterations = 2
default_ls_iterations = 5
default_model = 'g1_8dof_mjx_sphere'


def plot_batch_size_abalation():
    df = pd.read_csv('mjx_speed_test.csv')
    
    # filter df by model
    df = df[df['model'] == default_model]
    # filter df by batch size
    df = df[df['batch_size'] == default_batch_size]
    # filter df by iterations
    df = df[df['iterations'] == default_iterations]
    # filter df by ls_iterations
    df = df[df['ls_iterations'] == default_ls_iterations]

    # create two subplots, one for fps, one for single_realtime_factor
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.8)) 

    # set title
    fig.suptitle(f'{default_model} Batch Size Abalation')

    # plot fps with point markers
    sns.lineplot(data=df, x='batch_size', y='fps', ax=axs[0], marker='o')
    # label out the best performing batch size
    best_fps = df['fps'].max()
    best_batch_size = df[df['fps'] == best_fps]['batch_size'].values[0]
    axs[0].text(best_batch_size, best_fps, f'Best FPS: {best_fps:.1e} @ {best_batch_size}', ha='center', va='top')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('FPS')

    # plot single_realtime_factor
    sns.lineplot(data=df, x='batch_size', y='single_realtime_factor', ax=axs[1], marker='o')
    # label out the best performing batch size
    best_single_realtime_factor = df['single_realtime_factor'].max()
    best_batch_size = df[df['single_realtime_factor'] == best_single_realtime_factor]['batch_size'].values[0]
    axs[1].text(best_batch_size, best_single_realtime_factor, f'Best Single Realtime Factor: {best_single_realtime_factor:.1f} @ {best_batch_size}', ha='left', va='top')
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Batch Size')
    axs[1].set_ylabel('Single Realtime Factor')

    # save figure
    plt.tight_layout()
    plt.savefig('./results/batch_size_abalation.png', dpi=300)

def plot_iterations_abalation():
    df = pd.read_csv('mjx_speed_test.csv')

    # filter df by model
    df = df[df['model'] == default_model]
    # filter df by batch size
    df = df[df['batch_size'] == default_batch_size]
    # filter df by ls_iterations
    df = df[df['ls_iterations'] == default_ls_iterations]

    # create two subplots, one for fps, one for single_realtime_factor
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.8)) 

    # set title
    fig.suptitle(f'{default_model} Iterations Abalation')

    # plot fps with point markers
    sns.lineplot(data=df, x='iterations', y='fps', ax=axs[0], marker='o')
    # label out the best performing batch size
    best_fps = df['fps'].max()
    best_batch_size = df[df['fps'] == best_fps]['iterations'].values[0]
    axs[0].text(best_batch_size, best_fps, f'Best FPS: {best_fps:.1e} @ {best_batch_size}', ha='center', va='top')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('FPS')

    # plot single_realtime_factor
    sns.lineplot(data=df, x='iterations', y='single_realtime_factor', ax=axs[1], marker='o')
    # label out the best performing batch size
    best_single_realtime_factor = df['single_realtime_factor'].max()
    best_batch_size = df[df['single_realtime_factor'] == best_single_realtime_factor]['iterations'].values[0]
    axs[1].text(best_batch_size, best_single_realtime_factor, f'Best Single Realtime Factor: {best_single_realtime_factor:.1f} @ {best_batch_size}', ha='left', va='top')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Single Realtime Factor')

    # save figure
    plt.tight_layout()
    plt.savefig('./results/iterations_abalation.png', dpi=300)

def plot_ls_iterations_abalation():
    df = pd.read_csv('mjx_speed_test.csv')

    # filter df by model
    df = df[df['model'] == default_model]
    # filter df by batch size
    df = df[df['batch_size'] == default_batch_size]
    # filter df by iterations
    df = df[df['iterations'] == default_iterations]

    # create two subplots, one for fps, one for single_realtime_factor
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.8)) 

    # set title
    fig.suptitle(f'{default_model} Linsearch Iterations Abalation')

    # plot fps with point markers
    sns.lineplot(data=df, x='ls_iterations', y='fps', ax=axs[0], marker='o')
    # label out the best performing batch size
    best_fps = df['fps'].max()
    best_batch_size = df[df['fps'] == best_fps]['ls_iterations'].values[0]
    axs[0].text(best_batch_size, best_fps, f'Best FPS: {best_fps:.1e} @ {best_batch_size}', ha='center', va='top')
    axs[0].set_xlabel('Linsearch Iterations')
    axs[0].set_ylabel('FPS')

    # plot single_realtime_factor
    sns.lineplot(data=df, x='ls_iterations', y='single_realtime_factor', ax=axs[1], marker='o')
    # label out the best performing batch size
    best_single_realtime_factor = df['single_realtime_factor'].max()
    best_batch_size = df[df['single_realtime_factor'] == best_single_realtime_factor]['ls_iterations'].values[0]
    axs[1].text(best_batch_size, best_single_realtime_factor, f'Best Single Realtime Factor: {best_single_realtime_factor:.1f} @ {best_batch_size}', ha='left', va='top')
    axs[1].set_xlabel('Linsearch Iterations')
    axs[1].set_ylabel('Single Realtime Factor')

    # save figure
    plt.tight_layout()
    plt.savefig('./results/ls_iterations_abalation.png', dpi=300)

def plot_batch_size_abalation_over_model():
    df = pd.read_csv('mjx_speed_test.csv')
    models = ['g1_8dof_mjx_sphere', 'g1_8dof_mjx_capsule']
    # add '4090' to the end of batch_size
    df['batch_size'] = df['batch_size'].astype(str) + ' (4090)'

    # append mjc data
    mjc_df = pd.read_csv('mjc_speed_test.csv')
    mjc_df['batch_size'] = mjc_df['batch_size'].astype(str) + ' (7950X)'
    df = pd.concat([mjc_df,df])

    # create two subplots, one for fps, one for single_realtime_factor
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.8)) 

    # set title
    fig.suptitle(f'Batch Size Abalation Over Models')

    # plot fps with point markers
    sns.barplot(data=df, x='batch_size', y='fps', hue='model', ax=axs[0])
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('FPS')

    # plot single_realtime_factor
    sns.barplot(data=df, x='batch_size', y='single_realtime_factor', hue='model', ax=axs[1])
    # label out the best performing batch size
    axs[1].set_ylim(0, 50)
    axs[1].set_xlabel('Batch Size')
    axs[1].set_ylabel('Single Realtime Factor')

    # save figure
    plt.tight_layout()
    plt.savefig('./results/batch_size_abalation_over_model.png', dpi=300)

if __name__ == '__main__':
    plot_batch_size_abalation_over_model()