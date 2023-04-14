def focussing_efficiency(intensity1, intensity2):
    total_power = sum(intensity2)

    center = np.argmax(intensity1)
    length = max(np.size(intensity1))

    value = intensity1[center]
    for i in range(center+1, length):
        new_value = intensity1[i]
        if new_value > value:
            zero2 = i
        else:
            value = new_value

    value = intensity1[center]
    for i in range(center-1, 0, -1):
        new_value = intensity1[i]
        if new_value > value:
            zero1 = i
        else:
            value = new_value

    focussed_power = sum(intensity1[zero1+1:zero2])

    return focussed_power / total_power