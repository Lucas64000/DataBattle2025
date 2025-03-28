# DataBattle2025

Il faut executer les cellules contenant les %%writefile du notebook chatbot 
\n
Le fichier csv contient les consommations calculées avec codecarbone. La premiere ligne correspond à une requete au chatbot avec ollama utilisant le modèle gemma:7b, nous utilisons cependant l'api de mistral mais on pourra recalculer via le modèle mistral d'ollama ultérieurement étant donné que cest compliqué de calculer à cause de l'api. 
La derniere requete correspond à la consommation lors de la création des embeddings, et les autres la consommation pour générer les questions via gemma 7b. Il faut regarder la cellule 4 duration 581.052131359. On a oublié de stopper le tracker mais pour avoir la conso totale vu que c'est relativement proportionnel il faut multiplier par le nombre de noeuds / 100 (taille du batch) soit 500. 
