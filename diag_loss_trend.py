import json
with open('D:/NeuroCoder/logs/training_stats_partial.json') as f:
    data = json.load(f)
print('Total steps:', data['total_steps'])
for entry in data['loss_history'][-20:]:
    s = entry['step']
    l = entry['loss']
    ce = entry.get('ce_loss', 0)
    cr = entry.get('ce_raw', ce)
    aux = entry.get('aux_loss', 0)
    lr = entry.get('lr', 0)
    print(f'step {s:>5d}: loss={l:.4f} ce={ce:.4f} raw={cr:.4f} aux={aux:.4f} lr={lr:.6f}')
