from libc.math cimport fmod, pow
from libc.time cimport time
import numpy as np
cimport numpy as np
from array import array


cdef class Chromosome(object):
    # public will let python can access the attribute
    cdef public int np
    cdef public double f
    #cdef double[:] v
    cdef public object v
    #cdef public Chromosome *v

    def __cinit__(self, int n):
        self.np = n if n > 0 else 2
        self.f = 0.0
        self.v = [0.0] * self.np

    def cp(self, Chromosome obj):
        """
        copy all atribute from another chromsome object
        """
        self.np = obj.np
        self.f = obj.f
        self.v = obj.v[:]

    def get_v(self, int i):
        return self.v[i]

    def is_self(self, obj):
        """
        check the object is self?
        """
        return obj is self

    def assign(self, obj):
        if not self.is_self(obj):
            self.cp(obj)


cdef class Genetic(object):

    cdef int nParm, nPop, maxGen, gen, rpt
    cdef double pCross, pMute, pWin, bDelta, iseed, mask, seed
    cdef object func
    cdef object chrom, newChrom, babyChrom
    cdef Chromosome chromElite, chromBest
    cdef double[:] maxLimit, minLimit

    cdef int timeS, timeE
    cdef object fitnessTime, fitnessParameter

    def __cinit__(self, int nParm, int nPop, double pCross, double pMute, double pWin, double bDelta, upper, lower, objFunc):
        """
        init(function func)
        """
        # check nParm and list upper's len is equal
        if nParm != len(upper) or nParm != len(lower):
            raise Exception("nParm and upper's length and lower's length must be equal")
        self.func = objFunc
        self.nParm = nParm
        self.nPop = nPop
        self.pCross = pCross
        self.pMute = pMute
        self.pWin = pWin
        self.bDelta = bDelta

        self.chrom = [Chromosome(nParm) for i in range(nPop)]
        self.newChrom = [Chromosome(nParm) for i in range(nPop)]
        self.babyChrom = [Chromosome(nParm) for i in range(3)]

        self.chromElite = Chromosome(nParm)
        self.chromBest = Chromosome(nParm)

        self.maxLimit = array('d', upper[:])
        self.minLimit = array('d', lower[:])

        # maxgen and gen
        self.maxGen = 0
        self.gen = 0

        # base seed add mask
        self.seed = 0.0
        self.iseed = 470211272.0
        self.mask = int('7fffffff', 16)

        # setup benchmark
        self.timeS = time(NULL)
        self.timeE = 0
        self.fitnessTime = ''
        self.fitnessParameter = ''

    cdef void newSeed(self)except *:
        # if seed is 0, it mean not init with some random
        # use iseed assign
        if(self.seed == 0.0):
            self.seed = self.iseed
        else:
            self.seed = fmod(self.seed * 16807.0, self.mask)

    cdef double rnd(self)except *:
        self.newSeed()
        return self.seed/self.mask

    cdef void randomize(self)except *:
        self.seed = time(NULL)
        #self.seed = time.time()

    cdef int random(self, int k)except *:
        return int(self.rnd()*k)

    cdef double randVal(self, double low, double high)except *:
        return self.rnd()*(high-low)+low

    cdef double check(self, int i, double v)except *:
        """
        If a variable is out of bound,
        replace it with a random value
        """
        if (v > self.maxLimit[i]) or (v < self.minLimit[i]):
            return self.randVal(self.minLimit[i], self.maxLimit[i])
        return v

    cdef void crossOver(self)except *:
        cdef int i, s, j
        for i in range(0, self.nPop-1, 2):
            # crossover
            if(self.rnd() < self.pCross):
                for s in range(self.nParm):
                    # first baby, half father half mother
                    self.babyChrom[0].v[s] = 0.5 * self.chrom[i].v[s] + 0.5*self.chrom[i+1].v[s];
                    # second baby, three quaters of fater and quater of mother
                    self.babyChrom[1].v[s] = self.check(s, 1.5 * self.chrom[i].v[s] - 0.5*self.chrom[i+1].v[s])
                    # third baby, quater of fater and three quaters of mother
                    self.babyChrom[2].v[s] = self.check(s,-0.5 * self.chrom[i].v[s] + 1.5*self.chrom[i+1].v[s]);

                for j in range(3):
                    self.babyChrom[j].f = self.func(self.babyChrom[j].v)

                if self.babyChrom[1].f < self.babyChrom[0].f:
                    self.babyChrom[0], self.babyChrom[1] = self.babyChrom[1], self.babyChrom[0]

                if self.babyChrom[2].f < self.babyChrom[0].f:
                    self.babyChrom[2], self.babyChrom[0] = self.babyChrom[0], self.babyChrom[2]

                if self.babyChrom[2].f < self.babyChrom[1].f:
                    self.babyChrom[2], self.babyChrom[1] = self.babyChrom[1], self.babyChrom[2]

                # replace first two baby to parent, another one will be
                self.chrom[i].assign(self.babyChrom[0])
                self.chrom[i+1].assign(self.babyChrom[1])

    cdef double delta(self, double y)except *:
        cdef double r
        r = self.gen / self.maxGen
        return y*self.rnd()*pow(1.0-r, self.bDelta)

    cdef void fitness(self)except *:
        cdef int j
        for j in range(self.nPop):
            self.chrom[j].f = self.func(self.chrom[j].v)

        self.chromBest.assign(self.chrom[0])

        for j in range(1, self.nPop):
            if(self.chrom[j].f < self.chromBest.f):
                self.chromBest.assign(self.chrom[j])

        if(self.chromBest.f < self.chromElite.f):
            self.chromElite.assign(self.chromBest)

    cdef void initialPop(self)except *:
        cdef int i, j
        for j in range(self.nPop):
            for i in range(self.nParm):
                self.chrom[j].v[i] = self.randVal(self.minLimit[i], self.maxLimit[i])

    cdef void mutate(self)except *:
        cdef int i, s
        for i in range(self.nPop):
            if self.rnd() < self.pMute:
                s = self.random(self.nParm)
                if (self.random(2) == 0):
                    self.chrom[i].v[s] += self.delta(self.maxLimit[s]-self.chrom[i].v[s])
                else:
                    self.chrom[i].v[s] -= self.delta(self.chrom[i].v[s]-self.minLimit[s])

    cdef void report(self)except *:
        cdef int i
        cdef double p

        if self.gen == 0:
            print("Genetik results - Initial population")
        elif self.gen == self.maxGen:
            print("Final Genetik results at", self.gen, "generations")
        else:
            print("Genetik results after", self.gen, "generations")

        print("Function : %.6f" % (self.chromElite.f))
        for i, p in enumerate(self.chromElite.v):
            print("Var", i+1, ":", p)

    cdef void select(self)except *:
        """
        roulette wheel selection
        """
        cdef int i, j, k
        for i in range(self.nPop):
            j = self.random(self.nPop)
            k = self.random(self.nPop)
            self.newChrom[i].assign(self.chrom[j])
            if(self.chrom[k].f < self.chrom[j].f) and (self.rnd() < self.pWin):
                self.newChrom[i].assign(self.chrom[k])
        # in this stage, newChrom is select finish
        # now replace origin chrom
        for i in range(self.nPop):
            self.chrom[i].assign(self.newChrom[i])

        # select random one chrom to be best chrom, make best chrom still exist
        j = self.random(self.nPop);
        self.chrom[j].assign(self.chromElite)

    cdef void run(self, int mxg, int rp):
        """
        // **** Init and run GA for maxGen times
        // **** mxg : maximum generation
        // **** rp  : report cycle, 0 for final report or
        // ****       report each mxg modulo rp
        """
        self.maxGen = mxg
        self.rpt = rp

        self.randomize()
        self.initialPop()
        self.chrom[0].f = self.func(self.chrom[0].v)
        self.chromElite.assign(self.chrom[0])

        self.gen = 0
        self.fitness()
        self.report()

        for self.gen in range(1, self.maxGen + 1):
            self.select()
            self.crossOver()
            self.mutate()
            self.fitness()
            if self.rpt != 0:
                if self.gen % self.rpt == 0:
                    self.report()
        self.report()

