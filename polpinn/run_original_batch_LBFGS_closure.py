#  avec le closure 

def run_original_batch(params_pinns: dict, params: dict, S_f: DataAugmentation, S_j: DataAugmentation, output_path: Path):
    # === PHASE DE SETUP ===
    torch.manual_seed(1234)
    var_R = params_pinns["var_R"]
    batch_size = params_pinns["batch_size"]
    
    rayon_ini_norm, D_f_norm, ordre_R = normalisation(params["rayon_initialisation"], params["D_f"])
    params["ordre_R"] = ordre_R

    model = Physics_informed_nn(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=rayon_ini_norm,
        coeff_normal=params["P0_j"],
        var_R=var_R,
    )
    
    P0_f_norm = params["P0_f"] / params["P0_j"]
    F_f = Fick(D_f_norm, params["T_1"], P0_f_norm)
    
    print("Création du DataSet de points de collocation...")
    def_t = params["def_t"]; R_item = rayon_ini_norm
    nb_r_total, nb_t_total = 500, 500
    X_r_f_total = torch.linspace(0, R_item, nb_r_total).view(-1, 1)
    X_t_f_total = torch.linspace(0, def_t, nb_t_total).view(-1, 1)
    grid_r_f, grid_t_f = torch.meshgrid(X_r_f_total.squeeze(), X_t_f_total.squeeze(), indexing="ij")
    X_fick_total = torch.stack([grid_r_f.flatten(), grid_t_f.flatten()], dim=1)
    X_R_data_total = torch.full((nb_t_total, 1), R_item); X_t_data_total = torch.linspace(0, def_t, nb_t_total).view(-1, 1)
    X_boundary_total = torch.cat([X_R_data_total, X_t_data_total], dim=1)
    X_r_ini_total = torch.linspace(0, R_item, nb_t_total).view(-1, 1); X_t_ini_total = torch.zeros((nb_t_total, 1))
    X_ini_total = torch.cat([X_r_ini_total, X_t_ini_total], dim=1)
    X_data_total = torch.cat([X_boundary_total, X_ini_total], dim=0)
    print(f"DataSet créé: {X_fick_total.shape[0]} points de physique, {X_data_total.shape[0]} points de données.")

    loss = [[] for _ in range(5)]
    if var_R: loss.append([])
    model_opti = copy.deepcopy(model)
    min_loss_val = float('inf')

    # --- PHASE 1: ADAM AVEC MINI-BATCHS ---
    print("\n--- Phase 1: Adam Optimizer avec Mini-Batching ---")
    optimizer = optim.Adam(model.parameters(), lr=params_pinns['lr'])
    epochs_phase1 = 9000
    for it in tqdm(range(epochs_phase1), desc="Phase 1 (Adam)", file=sys.stdout):
        fick_indices = torch.randint(0, X_fick_total.shape[0], (batch_size // 2,))
        data_indices = torch.randint(0, X_data_total.shape[0], (batch_size // 2,))
        X_fick_batch = X_fick_total[fick_indices]; X_data_batch = X_data_total[data_indices]
        optimizer.zero_grad()
        L, L_total_list = cost_original_batch(model, F_f, S_f, S_j, X_fick_batch, X_data_batch)
        L.backward(); optimizer.step()
        if it % 10 == 0:
            for i in range(len(L_total_list)): loss[i].append(L_total_list[i])
            if var_R: loss[-1].append(model.R.item())
            if L_total_list[0] < min_loss_val: min_loss_val = L_total_list[0]; model_opti = copy.deepcopy(model)

    ### On repart du MEILLEUR modèle trouvé par Adam ###
    print(f"\nFin de la phase Adam. Meilleure perte trouvée : {min_loss_val:.2e}")
    print("Chargement du meilleur modèle pour L-BFGS...")
    model.load_state_dict(model_opti.state_dict())
    
    # ==============================================================================
    # --- PHASE 2: L-BFGS AVEC FULL-BATCH (NOUVELLE STRUCTURE)
    # ==============================================================================
    print("\n--- Phase 2: L-BFGS Optimizer (Nouvelle structure) ---")
    
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1.0,  # Le standard pour L-BFGS, car le line_search ajuste la taille du pas.
        max_iter=20,  # Nombre d'itérations par époque (standard)
        max_eval=None,
        tolerance_grad=1e-7, 
        tolerance_change=1e-9, 
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    # Dictionnaire pour garder les dernières métriques
    last_closure_metrics = {}
    epochs_phase2 = 100 # Nombre d'époques pour la phase L-BFGS

    for epoch in tqdm(range(epochs_phase2), desc="Phase 2 (L-BFGS)", file=sys.stdout):
        def closure():
            optimizer_lbfgs.zero_grad()
            L, L_total_list = cost_full_batch(model, F_f, S_f, S_j, X_fick_total, X_data_total)
            L.backward()
            
            # On stocke les métriques de cette évaluation pour le logging
            nonlocal last_closure_metrics
            last_closure_metrics['components'] = L_total_list
            if var_R:
                last_closure_metrics['R'] = model.R.item()
            return L
        
        # L'optimiseur exécute la closure plusieurs fois pour trouver le bon pas
        optimizer_lbfgs.step(closure)
        
        # --- Logging après chaque époque (chaque appel à step) ---
        # On récupère les métriques de la dernière évaluation réussie de la closure
        final_epoch_components = last_closure_metrics.get('components', [0]*5)
        for i in range(len(final_epoch_components)):
            loss[i].append(final_epoch_components[i])
        if var_R:
            loss[-1].append(last_closure_metrics.get('R', model.R.item()))
            
        # Mise à jour du meilleur modèle
        current_loss = final_epoch_components[0]
        if current_loss < min_loss_val:
            min_loss_val = current_loss
            model_opti = copy.deepcopy(model)

    print(f"\nEntraînement terminé. Meilleure perte finale (sum): {min_loss_val:.2e}")
    return model_opti, loss







# partire de méilleur modèle de adam

def run_original_batch(params_pinns: dict, params: dict, S_f: DataAugmentation, S_j: DataAugmentation, output_path: Path):
    # === MODIFIÉ POUR UN ENTRAÎNEMENT EN 2 PHASES (Adam -> L-BFGS) ===
    torch.manual_seed(1234)
    var_R = params_pinns["var_R"]
    batch_size = params_pinns["batch_size"]
    
    rayon_ini_norm, D_f_norm, ordre_R = normalisation(params["rayon_initialisation"], params["D_f"])
    params["ordre_R"] = ordre_R

    model = Physics_informed_nn(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=rayon_ini_norm,
        coeff_normal=params["P0_j"],
        var_R=var_R,
    )
    
    P0_f_norm = params["P0_f"] / params["P0_j"]
    F_f = Fick(D_f_norm, params["T_1"], P0_f_norm)
    
    # --- Création du DataSet complet (inchangé) ---
    print("Création du DataSet de points de collocation...")
    def_t = params["def_t"]; R_item = rayon_ini_norm
    nb_r_total, nb_t_total = 500, 500
    X_r_f_total = torch.linspace(0, R_item, nb_r_total).view(-1, 1)
    X_t_f_total = torch.linspace(0, def_t, nb_t_total).view(-1, 1)
    grid_r_f, grid_t_f = torch.meshgrid(X_r_f_total.squeeze(), X_t_f_total.squeeze(), indexing="ij")
    X_fick_total = torch.stack([grid_r_f.flatten(), grid_t_f.flatten()], dim=1)
    X_R_data_total = torch.full((nb_t_total, 1), R_item); X_t_data_total = torch.linspace(0, def_t, nb_t_total).view(-1, 1)
    X_boundary_total = torch.cat([X_R_data_total, X_t_data_total], dim=1)
    X_r_ini_total = torch.linspace(0, R_item, nb_t_total).view(-1, 1); X_t_ini_total = torch.zeros((nb_t_total, 1))
    X_ini_total = torch.cat([X_r_ini_total, X_t_ini_total], dim=1)
    X_data_total = torch.cat([X_boundary_total, X_ini_total], dim=0)
    print(f"DataSet créé: {X_fick_total.shape[0]} points de physique, {X_data_total.shape[0]} points de données.")

    loss = [[] for _ in range(5)]
    if var_R: loss.append([])

    model_opti = copy.deepcopy(model)
    min_loss_val = float('inf')

    # --- PHASE 1: ADAM AVEC MINI-BATCHS (inchangé) ---
    print("\n--- Phase 1: Adam Optimizer avec Mini-Batching ---")
    optimizer = optim.Adam(model.parameters(), lr=params_pinns['lr'])
    epochs_phase1 = 9000
    for it in tqdm(range(epochs_phase1), desc="Phase 1 (Adam)", file=sys.stdout):
        fick_indices = torch.randint(0, X_fick_total.shape[0], (batch_size // 2,))
        data_indices = torch.randint(0, X_data_total.shape[0], (batch_size // 2,))
        X_fick_batch = X_fick_total[fick_indices]; X_data_batch = X_data_total[data_indices]
        optimizer.zero_grad()
        L, L_total_list = cost_original_batch(model, F_f, S_f, S_j, X_fick_batch, X_data_batch)
        L.backward(); optimizer.step()
        if it % 10 == 0:
            for i in range(len(L_total_list)): loss[i].append(L_total_list[i])
            if var_R: loss[-1].append(model.R.item())
            if L_total_list[0] < min_loss_val: min_loss_val = L_total_list[0]; model_opti = copy.deepcopy(model)

    ### CORRECTION CRUCIALE : On repart du MEILLEUR modèle trouvé par Adam ###
    print(f"\nFin de la phase Adam. Meilleure perte trouvée : {min_loss_val:.2e}")
    print("Chargement du meilleur modèle pour L-BFGS...")
    model.load_state_dict(model_opti.state_dict())
    
    # --- PHASE 2: L-BFGS AVEC FULL-BATCH (NOUVELLE CONFIGURATION) ---
    print("\n--- Phase 2: L-BFGS Optimizer avec Full-Batch ---")
    
    # Nouvelle configuration de l'optimiseur :
    # - lr plus faible pour éviter le "saut de géant"
    # - max_iter beaucoup plus grand pour lui donner le temps de converger en une seule fois
    optimizer = optim.LBFGS(
        model.parameters(), 
        lr=0.1, 
        max_iter=500,  # Fait 500 itérations dans UN SEUL appel à step()
        max_eval=500*1.25,
        tolerance_grad=1e-7, 
        tolerance_change=1e-9, 
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    # On définit une liste temporaire pour les logs de cette phase
    lbfgs_loss_log = []

    def closure():
        optimizer.zero_grad()
        L, L_total_list = cost_full_batch(model, F_f, S_f, S_j, X_fick_total, X_data_total)
        L.backward()
        
        # On logue les pertes à chaque itération de L-BFGS
        lbfgs_loss_log.append(L_total_list)
        return L
    
    # On lance l'optimisation L-BFGS en UNE SEULE FOIS
    optimizer.step(closure)
    
    # On met à jour le modèle optimal et la perte minimale après la passe L-BFGS
    final_loss_lbfgs = lbfgs_loss_log[-1][0]
    if final_loss_lbfgs < min_loss_val:
        min_loss_val = final_loss_lbfgs
        model_opti = copy.deepcopy(model)

    # On ajoute les logs de la phase L-BFGS à l'historique principal
    for log_step in lbfgs_loss_log:
        for i in range(len(log_step)): loss[i].append(log_step[i])
        if var_R: loss[-1].append(model.R.item())

    print(f"\nEntraînement terminé. Meilleure perte finale (sum): {min_loss_val:.2e}")
    return model_opti, loss




#  10000 points de collocation et 200 pour les données 

# REMPLACEZ VOTRE ANCIENNE FONCTION run_original_batch PAR CELLE-CI

def run_original_batch(params_pinns: dict, params: dict, S_f: DataAugmentation, S_j: DataAugmentation, output_path: Path):
    # === PHASE DE SETUP (inchangée) ===
    torch.manual_seed(1234)
    var_R = params_pinns["var_R"]
    rayon_ini_norm, D_f_norm, ordre_R = normalisation(params["rayon_initialisation"], params["D_f"])
    params["ordre_R"] = ordre_R
    model = Physics_informed_nn(nb_layer=params_pinns["nb_hidden_layer"], hidden_layer=params_pinns["nb_hidden_perceptron"], rayon_ini=rayon_ini_norm, coeff_normal=params["P0_j"], var_R=var_R)
    P0_f_norm = params["P0_f"] / params["P0_j"]
    F_f = Fick(D_f_norm, params["T_1"], P0_f_norm)
    
    # ==============================================================================
    # --- MODIFICATIONS POUR UN TEST RAPIDE ---
    # ==============================================================================
    print("Création du DataSet de points de collocation (VERSION RAPIDE)...")
    def_t = params["def_t"]; R_item = rayon_ini_norm
    
    # 1. Réduction du nombre de points
    nb_r_total = 100  # sqrt(10000)
    nb_t_total = 100  # sqrt(10000)
    
    # 2. Réduction du nombre de points de données
    nb_t_data = 100 

    # --- Création du DataSet avec les nouvelles tailles ---
    X_r_f_total = torch.linspace(0, R_item, nb_r_total).view(-1, 1)
    X_t_f_total = torch.linspace(0, def_t, nb_t_total).view(-1, 1)
    grid_r_f, grid_t_f = torch.meshgrid(X_r_f_total.squeeze(), X_t_f_total.squeeze(), indexing="ij")
    X_fick_total = torch.stack([grid_r_f.flatten(), grid_t_f.flatten()], dim=1) # Total = 10000 points
    
    X_R_data_total = torch.full((nb_t_data, 1), R_item); X_t_data_total = torch.linspace(0, def_t, nb_t_data).view(-1, 1)
    X_boundary_total = torch.cat([X_R_data_total, X_t_data_total], dim=1)
    X_r_ini_total = torch.linspace(0, R_item, nb_t_data).view(-1, 1); X_t_ini_total = torch.zeros((nb_t_data, 1))
    X_ini_total = torch.cat([X_r_ini_total, X_t_ini_total], dim=1)
    X_data_total = torch.cat([X_boundary_total, X_ini_total], dim=0) # Total = 200 points
    
    print(f"DataSet créé: {X_fick_total.shape[0]} points de physique, {X_data_total.shape[0]} points de données.")

    loss = [[] for _ in range(5)]
    if var_R: loss.append([])
    model_opti = copy.deepcopy(model)
    min_loss_val = float('inf')

    # --- PHASE 1: ADAM - NOUVELLE LOGIQUE DE MINI-BATCHING COMPLET ---
    print("\n--- Phase 1: Adam Optimizer (Mini-Batching Complet) ---")
    optimizer = optim.Adam(model.parameters(), lr=params_pinns['lr'])
    epochs_phase1 = 10 # 10 époques pour le test
    
    # 3. Définition des tailles de batch
    fick_batch_size = 1000
    data_batch_size = 100 # J'ai mis 100 au lieu de 10 car 10 est très petit et vous avez 200 points de données.
    
    # Création des DataLoaders pour gérer les batchs facilement
    fick_dataset = torch.utils.data.TensorDataset(X_fick_total)
    fick_loader = torch.utils.data.DataLoader(fick_dataset, batch_size=fick_batch_size, shuffle=True)
    data_dataset = torch.utils.data.TensorDataset(X_data_total)
    data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=data_batch_size, shuffle=True)

    for epoch in range(epochs_phase1):
        epoch_loss_sum = 0.0
        # On itère sur les deux loaders en parallèle
        # zip s'arrêtera quand le plus petit loader sera épuisé (celui des données)
        for (fick_batch,), (data_batch,) in tqdm(zip(fick_loader, data_loader), desc=f"Epoch {epoch+1}/{epochs_phase1} (Adam)", file=sys.stdout):
            optimizer.zero_grad()
            L, L_total_list = cost_original_batch(model, F_f, S_f, S_j, fick_batch, data_batch)
            L.backward(); optimizer.step()
            epoch_loss_sum += L_total_list[0]

        # Logging à la fin de chaque époque complète
        avg_epoch_loss = epoch_loss_sum / len(data_loader)
        if avg_epoch_loss < min_loss_val: 
            min_loss_val = avg_epoch_loss
            model_opti = copy.deepcopy(model)
        
        # On log les composantes du dernier batch pour le graphique
        for i in range(len(L_total_list)): loss[i].append(L_total_list[i])
        if var_R: loss[-1].append(model.R.item())

    ### On repart du MEILLEUR modèle (inchangé) ###
    print(f"\nFin de la phase Adam. Meilleure perte trouvée : {min_loss_val:.2e}")
    print("Chargement du meilleur modèle pour L-BFGS...")
    model.load_state_dict(model_opti.state_dict())
    
    # --- PHASE 2: L-BFGS - TEST RAPIDE ---
    print("\n--- Phase 2: L-BFGS Optimizer (Test Rapide) ---")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    last_closure_metrics = {}
    epochs_phase2 = 10 # 10 époques pour le test

    for epoch in tqdm(range(epochs_phase2), desc="Phase 2 (L-BFGS)", file=sys.stdout):
        def closure():
            optimizer_lbfgs.zero_grad()
            L, L_total_list = cost_full_batch(model, F_f, S_f, S_j, X_fick_total, X_data_total)
            L.backward()
            nonlocal last_closure_metrics
            last_closure_metrics['components'] = L_total_list
            if var_R: last_closure_metrics['R'] = model.R.item()
            return L
        
        optimizer_lbfgs.step(closure)
        final_epoch_components = last_closure_metrics.get('components', [0]*5)
        for i in range(len(final_epoch_components)): loss[i].append(final_epoch_components[i])
        if var_R: loss[-1].append(last_closure_metrics.get('R', model.R.item()))
        current_loss = final_epoch_components[0]
        if current_loss < min_loss_val:
            min_loss_val = current_loss
            model_opti = copy.deepcopy(model)

    print(f"\nEntraînement terminé. Meilleure perte finale (sum): {min_loss_val:.2e}")
    return model_opti, loss