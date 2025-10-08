import numpy as np

# --- Dados de exemplo ---
transformation_matrix = np.random.rand(3, 3, 2, 2)

# Calcula o tensor auxiliar (índices: a, c, b, d, ...)
auxiliar_tensor = np.einsum("ab...,cd...->acbd...", transformation_matrix, transformation_matrix)
print("Shape original:", auxiliar_tensor.shape)  # (3, 3, 3, 3, 2, 2)

# --- Transformação correta ---
shape_rest = auxiliar_tensor.shape[4:]
# não há necessidade de transpor — já está (a, c, b, d)
aux = auxiliar_tensor.reshape(9, 9, *shape_rest)
print("Shape transformado:", aux.shape)  # (9, 9, 2, 2)

# --- Verificação ---
ok = True
for _ in range(5):
    a, b, c, d = np.random.randint(0, 3, 4)
    i = 3 * a + c  # linha (a,c)
    j = 3 * b + d  # coluna (b,d)

    diff = np.abs(aux[i, j] - auxiliar_tensor[a, c, b, d])
    if not np.allclose(diff, 0):
        ok = False
        print(f"Erro em (a,b,c,d)=({a},{b},{c},{d}) -> diferença máxima: {diff.max()}")

if ok:
    print("✅ Verificação concluída: todos os elementos correspondem corretamente.")
