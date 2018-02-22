last_img = x[-1][0][0][0].data
last = int(last_img.cpu().numpy()[0])
x = x[:-1]

indexes = indexes[last]