import torch

def do_simulation(x0, xdot0, dt, tf, p):
    xcur = torch.tensor(x0, requires_grad=True, dtype=torch.float64)  # 设置 requires_grad=True
    xpre = xcur - dt * xdot0
    x_lst = [x0]

    for i in range(tf):
        xnext = - (xcur * (-2 - dt * p[0] + p[1] * dt ** 2) + xpre + dt ** 2) / (1 + dt * p[0])

        x_lst.append(xnext.item())

        xpre = xcur
        xcur = xnext
    
    return x_lst, xcur

if __name__ == "__main__":
    tar_pos = -2.0
    p = torch.tensor([0.3, 0.1], requires_grad=True, dtype=torch.float64)  # 设置 requires_grad=True
    max_iters = 100000
    output_iter = 100
    dt = 1e-2
    target_timept = 0.5
    tf = int(target_timept / dt)
    x0 = 0
    xdot0 = 0
    alpha = 1

    optimizer = torch.optim.SGD([p], lr=alpha)

    for iter in range(max_iters):
        x_lst, x_final = do_simulation(x0, xdot0, dt, tf, p)

        loss = (x_final - tar_pos) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        should_stop = loss < 1e-6
        should_output = should_stop == True or iter % output_iter == 0
        if should_output:
            print(f"iter {iter} cur p {p[0].item():.3f} {p[1].item():.3f} x_final {x_final.item():.3f} energy {loss.item():.5f}")
        if should_stop:
            print(f"iter {iter}")
            print(f"optimization done! x_final {x_final.item():.3f} ~= target pos {tar_pos}, current energy is {loss.item():.3e}")
            break
