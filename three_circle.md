Exact solution:
$$
    u_{\text{exact}} (x, y) = \begin{cases}
        L(x, y) - C_{r_1}, \quad &L_1 (x, y) < 0, \\
        L(x, y) - C_{r_2}, \quad &L_2 (x, y) < 0, \\
        L(x, y) - C_{r_3}, \quad &L_3 (x, y) < 0, \\
        L(x, y), \quad &\text{otherwise}
    \end{cases}
$$
Where $C_{r_i} = [u]_{\Gamma_i} = \alpha_i \nabla u|_{\Omega^+} \cdot \mathbf{n}^+$, $L(x,y) = L_1 (x,y)L_2 (x,y)L_3 (x,y)$, and $[\partial_{\mathbf{n}} u]_{\Gamma_i} = 0$.

In another, 
$$
\begin{align*}
    L_1 (x, y) &= (x - 0.5)^2 + (y - 0.5)^2 - r_1^2, \quad r_1 = 0.25, \\
    L_2 (x, y) &= (x + 0.5)^2 + (y - 0.5)^2 - r_2^2, \quad r_2 = 0.30, \\
    L_1 (x, y) &= (x - 0.5)^2 + (y + 0.5)^2 - r_3^2, \quad r_3 = 0.20
\end{align*}
$$