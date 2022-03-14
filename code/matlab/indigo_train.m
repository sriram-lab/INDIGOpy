function [train_interactions, trainxnscores,phenotype_labels, indigo_model, sigma_delta_scores,conditions] = indigo_train(interaction_filename,annotation_filename,chemogenomics_filename,z,phenotype_data, phenotype_labels, conditions,interaction_scores,interaction_pairs)
% [train_interactions, trainxnscores,phenotype_labels, indigo_model, sigma_delta_scores,conditions] = indigo_train(interaction_filename,annotation_filename,chemogenomics_filename,z)
%%%% steps%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 load drug interaction data
% 2. convert  and match drug interaction data labels with chemogenomic data
% 3 input to indigo and train model
%%%%%%%%%%% input processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('z','var') || isempty(z)
    z = 2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 a load drug interaction data
if ~exist('interaction_scores','var') || isempty(interaction_scores)
[interaction_scores, interaction_pairs] = xlsread(interaction_filename);
end
drugs_all = unique(interaction_pairs);
%disp(drugs_all)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2a. match drugs with identifiers in chemogenomic data. 
%  convert  and match drug interaction data labels with chemical genetic data labels
 [num, txt] = xlsread(annotation_filename);
 [drugxn_id, chemgen_id] = deal(txt(:,1),txt(:,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drugnames_cell = drugs_all;
for i = 1:length(drugxn_id)
    drugnames_cell (ismember(drugnames_cell,drugxn_id(i))) = chemgen_id(i);
end

drugpairsname_cell = interaction_pairs;
for i = 1:length(drugxn_id)
    drugpairsname_cell(ismember(drugpairsname_cell,drugxn_id(i))) = chemgen_id(i);
end
drug_abb_chem = drugs_all(~ismember(drugs_all,drugnames_cell)); %these drugs have chem gen data
% disp(drug_abb_chem)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2b. load and match with chemical genetic data
if ~exist('phenotype_data','var') || isempty(phenotype_data)
[phenotype_data, phenotype_labels, conditions] = process_chemgen(chemogenomics_filename,z);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ix = ismember(drugpairsname_cell,conditions); 
ix = (ix(:,1) & ix(:,2)); 
train_interactions = drugpairsname_cell(ix,:); %% these xns have chemogenomic data
trainxnscores = interaction_scores(ix);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
traindrugs = unique(train_interactions(:));
[ix, pos] = ismember(traindrugs,conditions); 
trainchemgen = phenotype_data(:,pos);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4 input to indigo
% disp(train_interactions)
% disp(traindrugs)
% size(trainchemgen)
% size(trainxnscores)
[testinteractions_scores, indigo_model, sigma_delta_scores]  = indigo_rf(traindrugs,trainchemgen, train_interactions, trainxnscores, [],[], [],1,[]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
end