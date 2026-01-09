from exp import *


def count_wins(
    data: str,
    seed_1_time: str,
    seed_42_time: str,
    seed_124_time: str,
    seed_256_time: str,
    seed_369_time: str,
) -> None:
    subdata_list = []
    
    carla_seed_1_f1_list = []
    carla_seed_1_auc_list = []
    exp_seed_1_f1_list = []
    exp_seed_1_auc_list = []

    carla_seed_42_f1_list = []
    carla_seed_42_auc_list = []
    exp_seed_42_f1_list = []
    exp_seed_42_auc_list = []

    carla_seed_124_f1_list = []
    carla_seed_124_auc_list = []
    exp_seed_124_f1_list = []
    exp_seed_124_auc_list = []

    carla_seed_256_f1_list = []
    carla_seed_256_auc_list = []
    exp_seed_256_f1_list = []
    exp_seed_256_auc_list = []

    carla_seed_369_f1_list = []
    carla_seed_369_auc_list = []
    exp_seed_369_f1_list = []
    exp_seed_369_auc_list = []

    carla_seed_1_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'carla', f'seed_1.log'
    )
    carla_seed_42_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'carla', f'seed_42.log'
    )
    carla_seed_124_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'carla', f'seed_124.log'
    )
    carla_seed_256_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'carla', f'seed_256.log'
    )
    carla_seed_369_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'carla', f'seed_369.log'
    )
    
    exp_seed_1_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'exp', f'{seed_1_time}.log'
    )
    exp_seed_42_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'exp', f'{seed_42_time}.log'
    )
    exp_seed_124_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'exp', f'{seed_124_time}.log'
    )
    exp_seed_256_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'exp', f'{seed_256_time}.log'
    )
    exp_seed_369_log_dir = os.path.join(
        'log', 'pretext_classification', data, 'exp', f'{seed_369_time}.log'
    )

    with open(carla_seed_1_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'inference' in content:
                content_list = content.split(' ')
                subdata_list.append(content_list[-1][:-3])
            elif 'Best F1 score' in content:
                content_list = content.split(' ')
                carla_seed_1_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                carla_seed_1_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(carla_seed_42_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Best F1 score' in content:
                content_list = content.split(' ')
                carla_seed_42_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                carla_seed_42_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(carla_seed_124_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Best F1 score' in content:
                content_list = content.split(' ')
                carla_seed_124_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                carla_seed_124_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(carla_seed_256_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Best F1 score' in content:
                content_list = content.split(' ')
                carla_seed_256_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                carla_seed_256_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(carla_seed_369_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Best F1 score' in content:
                content_list = content.split(' ')
                carla_seed_369_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                carla_seed_369_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(exp_seed_1_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Seed' in content:
                content_list = content.split(' ')
                exp_seed = int(content_list[-1])
                assert(exp_seed == 1), \
                    f"exp seed {exp_seed} and carla seed 1 mismatch."
            elif 'Best F1 score' in content:
                content_list = content.split(' ')
                exp_seed_1_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                exp_seed_1_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(exp_seed_42_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Seed' in content:
                content_list = content.split(' ')
                exp_seed = int(content_list[-1])
                assert(exp_seed == 42), \
                    f"exp seed {exp_seed} and carla seed 42 mismatch."
            elif 'Best F1 score' in content:
                content_list = content.split(' ')
                exp_seed_42_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                exp_seed_42_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(exp_seed_124_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Seed' in content:
                content_list = content.split(' ')
                exp_seed = int(content_list[-1])
                assert(exp_seed == 124), \
                    f"exp seed {exp_seed} and carla seed 124 mismatch."
            elif 'Best F1 score' in content:
                content_list = content.split(' ')
                exp_seed_124_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                exp_seed_124_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(exp_seed_256_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Seed' in content:
                content_list = content.split(' ')
                exp_seed = int(content_list[-1])
                assert(exp_seed == 256), \
                    f"exp seed {exp_seed} and carla seed 256 mismatch."
            elif 'Best F1 score' in content:
                content_list = content.split(' ')
                exp_seed_256_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                exp_seed_256_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    with open(exp_seed_369_log_dir, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if 'Seed' in content:
                content_list = content.split(' ')
                exp_seed = int(content_list[-1])
                assert(exp_seed == 369), \
                    f"exp seed {exp_seed} and carla seed 369 mismatch."
            elif 'Best F1 score' in content:
                content_list = content.split(' ')
                exp_seed_369_f1_list.append(float(content_list[-1]))
            elif 'AUC-PR' in content:
                content_list = content.split(' ')
                exp_seed_369_auc_list.append(float(content_list[-1]))
            elif 'Scores' in content:
                break
    
    subdata_list = np.array(subdata_list).reshape(-1)

    carla_seed_1_f1_list = np.array(carla_seed_1_f1_list)
    exp_seed_1_f1_list = np.array(exp_seed_1_f1_list)
    carla_seed_1_auc_list = np.array(carla_seed_1_auc_list)
    exp_seed_1_auc_list = np.array(exp_seed_1_auc_list)

    carla_seed_42_f1_list = np.array(carla_seed_42_f1_list)
    exp_seed_42_f1_list = np.array(exp_seed_42_f1_list)
    carla_seed_42_auc_list = np.array(carla_seed_42_auc_list)
    exp_seed_42_auc_list = np.array(exp_seed_42_auc_list)

    carla_seed_124_f1_list = np.array(carla_seed_124_f1_list)
    exp_seed_124_f1_list = np.array(exp_seed_124_f1_list)
    carla_seed_124_auc_list = np.array(carla_seed_124_auc_list)
    exp_seed_124_auc_list = np.array(exp_seed_124_auc_list)

    carla_seed_256_f1_list = np.array(carla_seed_256_f1_list)
    exp_seed_256_f1_list = np.array(exp_seed_256_f1_list)
    carla_seed_256_auc_list = np.array(carla_seed_256_auc_list)
    exp_seed_256_auc_list = np.array(exp_seed_256_auc_list)

    carla_seed_369_f1_list = np.array(carla_seed_369_f1_list)
    exp_seed_369_f1_list = np.array(exp_seed_369_f1_list)
    carla_seed_369_auc_list = np.array(carla_seed_369_auc_list)
    exp_seed_369_auc_list = np.array(exp_seed_369_auc_list)

    seed_1_f1_wins = carla_seed_1_f1_list < exp_seed_1_f1_list
    seed_1_auc_wins = carla_seed_1_auc_list < exp_seed_1_auc_list
    seed_1_all_wins = subdata_list[seed_1_f1_wins & seed_1_auc_wins]

    seed_42_f1_wins = carla_seed_42_f1_list < exp_seed_42_f1_list
    seed_42_auc_wins = carla_seed_42_auc_list < exp_seed_42_auc_list
    seed_42_all_wins = subdata_list[seed_42_f1_wins & seed_42_auc_wins]

    seed_124_f1_wins = carla_seed_124_f1_list < exp_seed_124_f1_list
    seed_124_auc_wins = carla_seed_124_auc_list < exp_seed_124_auc_list
    seed_124_all_wins = subdata_list[seed_124_f1_wins & seed_124_auc_wins]

    seed_256_f1_wins = carla_seed_256_f1_list < exp_seed_256_f1_list
    seed_256_auc_wins = carla_seed_256_auc_list < exp_seed_256_auc_list
    seed_256_all_wins = subdata_list[seed_256_f1_wins & seed_256_auc_wins]

    seed_369_f1_wins = carla_seed_369_f1_list < exp_seed_369_f1_list
    seed_369_auc_wins = carla_seed_369_auc_list < exp_seed_369_auc_list
    seed_369_all_wins = subdata_list[seed_369_f1_wins & seed_369_auc_wins]

    all_win_count = np.array([0 for _ in range(len(subdata_list))])
    
    for all_win in seed_1_all_wins:
        all_win_idx = np.argwhere(subdata_list == all_win)
        all_win_count[all_win_idx] += 1
    
    for all_win in seed_42_all_wins:
        all_win_idx = np.argwhere(subdata_list == all_win)
        all_win_count[all_win_idx] += 1

    for all_win in seed_124_all_wins:
        all_win_idx = np.argwhere(subdata_list == all_win)
        all_win_count[all_win_idx] += 1
    
    for all_win in seed_256_all_wins:
        all_win_idx = np.argwhere(subdata_list == all_win)
        all_win_count[all_win_idx] += 1
    
    for all_win in seed_369_all_wins:
        all_win_idx = np.argwhere(subdata_list == all_win)
        all_win_count[all_win_idx] += 1
    
    win_subdata_list = []
    double_win_subdata_list = []
    triple_win_subdata_list = []
    quadra_win_subdata_list = []
    penta_win_subdata_list = []

    for i in range(len(all_win_count)):
        if all_win_count[i] == 1:
            win_subdata_list.append(subdata_list[i])
        elif all_win_count[i] == 2:
            double_win_subdata_list.append(subdata_list[i])
        elif all_win_count[i] == 3:
            triple_win_subdata_list.append(subdata_list[i])
        elif all_win_count[i] == 4:
            quadra_win_subdata_list.append(subdata_list[i])
        elif all_win_count[i] == 5:
            penta_win_subdata_list.append(subdata_list[i]) 
    
    print(f'Win counts:')
    print(f'1: {win_subdata_list}')
    print(f'2: {double_win_subdata_list}')
    print(f'3: {triple_win_subdata_list}')
    print(f'4: {quadra_win_subdata_list}')
    print(f'5: {penta_win_subdata_list}')
    
    return


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--data',
        type=str,
    )
    args.add_argument(
        '--seed-1-time',
        type=str,
    )
    args.add_argument(
        '--seed-42-time',
        type=str,
    )
    args.add_argument(
        '--seed-124-time',
        type=str,
    )
    args.add_argument(
        '--seed-256-time',
        type=str,
    )
    args.add_argument(
        '--seed-369-time',
        type=str,
    )
    config = args.parse_args()
    count_wins(
        data=config.data,
        seed_1_time=config.seed_1_time,
        seed_42_time=config.seed_42_time,
        seed_124_time=config.seed_124_time,
        seed_256_time=config.seed_256_time,
        seed_369_time=config.seed_369_time,
    )
