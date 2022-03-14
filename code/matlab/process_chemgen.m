function [phenotype_data, phenotype_labels, conditions] = process_chemgen(fname,z)
%[phenotype_data, phenotype_labels, conditions] = process_chemgen(filename)
% processes chemogenomic data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load data
if isempty(fname)
    fname = 'ecoli_phenotype_data_cell.xlsx';
end
 [phenotype_num, txt] = xlsread(fname); % supplementary file. nicholsetal
probelist = txt(3:end,1);  conditions = txt(2,2:end)';


if ~exist('z','var') || isempty(z)
    z = 2;
end
%disp(z)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% converting gene ids to standard ids
load ecoli_annotation_data1 genenames_array      genenames_array1    genenames_bnums

clear plist
plist = cell(size(probelist));
for i = 1:length(probelist)
    tem = regexp(probelist{i},'-','split');
    try
        plist(i) = tem(2);
    catch
        plist(i) = tem(1);
    end
end
plist = regexprep(plist,'''','');

[ix, pos] =ismember(upper(plist),upper(genenames_array));
plist_bnums = plist;
plist_bnums(ix) = genenames_bnums(pos(ix));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% normalizing data
%phenotype_num = quantilenorm(phenotype_num);
%z = 2;
for i = 1:324,
    te = plist_bnums(phenotype_num(:,i) < -z);
    cell_z1_list_t(1:length(te),i) = te;
    lte(i) = length(te);
end
cell_z1_list_t = regexprep(cell_z1_list_t,'''','');
phenotype_labels = unique(cell_z1_list_t(:));

clear nicholslistix_t
for i = 1:324,
    nicholslistix_t(:,i) = ismember(phenotype_labels,cell_z1_list_t(:,i));
end
phenotype_data = nicholslistix_t;

end
