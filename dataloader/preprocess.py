import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from visualizer.plots import plot_genotype_hist

rng = np.random.RandomState(0)

def g(file):
    with open(file,"r") as f:
        for line in f.readlines():
            for item in list(map(int, line.strip("\n"))):
                yield item

class Preprocesser:
    def __init__(self,
                 filebase: str,
                 normalization: bool = True,
                 impute_missing: str = "MF",
                 train_val_split: float = 0.2,
                 missing_val_ori: float = 9.0,
                 missing_val_norm: float = -1.0,
                 split_policy: str = "random"):
        """
        @param filebase: path+ filename without suffix eg. "/home/data/TPS_full/new_TPS"
        @param normalization: wheter to normalize input
        @param impute_missing: method for missing value imputation, no imputation if None
        @param train_val_split: fraction of validation set
        @param missing_val_ori: missing value in input
        @param missing_val_norm: missing value after normalization
        """
        self.filebase = filebase
        self.normalization = normalization
        self.impute_missing = impute_missing
        self.split = train_val_split
        self.missing_val_ori = missing_val_ori
        self.missing_val_norm = missing_val_norm
        self.split_policy = split_policy
        self.inds, self.targets = self._load_inds()
        self.genotypes = self._load_genotype()
        self.missing_mask = (self.genotypes!=self.missing_val_ori).astype(int).T # 0 means missing
        print(f"Overall missing rate is {1-self.missing_mask.sum()/self.missing_mask.size}")

    def _load_inds(self):
        """
        @return: individual and corresponding population
        @rtype: np.array in shape of (num_inds, 2), col1 is ID, col2 is population
        """
        ind_pop_list = np.genfromtxt(self.filebase + ".ind", usecols=(0, 2), dtype=str)
        print("Reading individual profiles from " + self.filebase + ".ind")
        targets = np.genfromtxt(self.filebase + ".ind", usecols=(3,4,5,6), dtype=float)
        print("Reading targets from " + self.filebase + ".ind")
        return ind_pop_list, targets

    def _load_genotype(self):
        """
        @return: raw genotype data in 0,1,2,9(missing)
        @rtype: np.array in shape of (num_SNPs, num_inds)
        """
        num_inds = len(self.inds)
        genotypes = np.fromiter(g(self.filebase + ".eigenstratgeno"), dtype=float).reshape(-1, num_inds)
        print("Reading genotypes from " + self.filebase + ".eigenstratgeno")
        print("genotypes's shape", genotypes.shape)
        return genotypes

    def _normalize(self):
        """
        Normalize genotypes into interval [0,1] by translating 0,1,2 -> 0.0, 0.5, 1.0, missing value (default 9) -> -1
        """
        self.genotypes[self.genotypes == 1.0] = 0.5
        self.genotypes[self.genotypes == 2.0] = 1.0
        self.genotypes[self.genotypes == self.missing_val_ori] = self.missing_val_norm

    def _impute_missing(self):
        if self.impute_missing == "MF":
            self._most_frequent_impute()
            print("Imputed missing value using most frequent allele")
        elif self.impute_missing == "AE":
            self._AE_impute()
            print("Imputed missing value using AE reconstruction")
        else:
            raise NotImplementedError(f"{self.impute_missing} is not implemented")

    def _most_frequent_impute(self):
        for m in self.genotypes:
            # exclude missing value before count modes
            modes = stats.mode(m[m!=self.missing_val_norm])
            most_frequent = modes.mode
            if most_frequent.size ==0:
                raise Warning("Run into SNPs with 100% missing rate")
            else:
                m[m==self.missing_val_norm]=most_frequent[0]

    def _AE_impute(self):
        # Reconstructed genotype file should be placed at the same folder as .eigenstratgeno file
        # and with suffix .npy
        reconstruct= np.load(self.filebase+".npy", dtype=float)
        print ("load reconstructed genotype from "+ self.filebase+".npy")
        assert reconstruct.shape == self.genotypes
        # normalize before autoencoder imputation
        if not self.normalization:
            print("Normalization before AE")
            self._normalize()
        self.genotypes = self.genotypes * self.missing_mask + reconstruct * (1-self.missing_mask)


    def _train_val_split(self):
        if self.split_policy == "random":
            geno_train, geno_val, inds_train, inds_val, mask_train, mask_val= train_test_split(self.genotypes, self.inds,
                                                                                               self.missing_mask,
                                                                                               test_size=self.split,
                                                                                               random_state=rng)
        elif self.split_policy == "stratified":
            pop_list=self.inds[:,1]
            pops, pop_counts = np.unique(pop_list,return_counts=True)
            rare_pops = pops[pop_counts==1]
            for rare in rare_pops:
                pop_list[pop_list==rare]="Rare_pop"

            geno_train, geno_val, inds_train, inds_val, mask_train, mask_val = train_test_split(self.genotypes, self.inds,
                                                                                                self.missing_mask,
                                                                                                test_size=self.split,
                                                                                                random_state=rng,
                                                                                                stratify=pop_list)
        else:
            raise NotImplementedError(f"{self.split_policy} is not implemented" )
        return geno_train, geno_val, inds_train, inds_val, mask_train, mask_val

    def _save(self):
        return

    def process(self):
        """
        input should be raw genotype data in 0,1,2,9(missing)
        """
        if self.normalization:
            self._normalize()
            print("Normalized genome by mapping 0 to 0, 1 to 0.5, 2 to 1.0")
        if self.impute_missing is not None:
            self._impute_missing()
        #     transpose genotype data into shape of (n_inds, n_SNPs)
        self.genotypes = self.genotypes.T
        plot_genotype_hist(self.genotypes,self.filebase+"_genotype_hist")
        geno_train, geno_val, inds_train, inds_val, mask_train, mask_val = self._train_val_split()
        # save data
        data_dict = {"genotype":{"all":self.genotypes,"train":geno_train,"val":geno_val},
                     "inds":{"all":self.inds,"train":inds_train,"val":inds_val},
                     "mask":{"all":self.missing_mask,"train":mask_train,"val":mask_val}}
        for dtyp , v in data_dict.items():
            for split, value in v.items():
                if dtyp == "genotype":
                    np.save(self.filebase+f"_{self.impute_missing}_{dtyp}_{split}.npy",value.astype(np.float16))
                elif dtyp == "inds":
                    np.save(self.filebase+f"_{self.impute_missing}_{dtyp}_{split}.npy",value)
                elif dtyp =="mask":
                    np.save(self.filebase+f"_{self.impute_missing}_{dtyp}_{split}.npy",value.astype(int))
                else:
                    raise ValueError(f"Wrong data {dtyp}")
        np.save(self.filebase+f"_{self.impute_missing}_Y_all.npy", self.targets)


if __name__ == "__main__":
    params= {"filebase":"/home/etlar/daqu/Projects/genome/data/aDNA/new_data_1Feb_Daqu/TPS2_slim/TPS2_mind5_ex_2233modern_maf05",
             "normalization": True,
             "impute_missing": "MF",
             "train_val_split": 0.2,
             "missing_val_ori": 9.0,
             "split_policy": "stratified"}
    processer = Preprocesser(**params)
    processer.process()