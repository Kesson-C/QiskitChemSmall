import numpy as np
from scipy import sparse
from pyquante2 import molecule,basisset,rhf,uhf
from pyquante2.ints.integrals import onee_integrals,twoe_integrals
from pyquante2.utils import simx

# hpq and hpqrs functions are from qiskit
def onee_to_spin(mohij, mohij_b=None, threshold=1E-12):
    if mohij_b is None:
        mohij_b = mohij
        
    # The number of spin orbitals is twice the number of orbitals
    norbs = mohij.shape[0]
    nspin_orbs = 2*norbs
    # One electron terms
    moh1_qubit = np.zeros([nspin_orbs, nspin_orbs])
    for p in range(nspin_orbs):  # pylint: disable=invalid-name
        for q in range(nspin_orbs):
            spinp = int(p/norbs)
            spinq = int(q/norbs)
            if spinp % 2 != spinq % 2:
                continue
            ints = mohij if spinp == 0 else mohij_b
            orbp = int(p % norbs)
            orbq = int(q % norbs)
            if abs(ints[orbp, orbq]) > threshold:
                moh1_qubit[p, q] = ints[orbp, orbq]
    return moh1_qubit

def twoe_to_spin(mohijkl, mohijkl_bb=None, mohijkl_ba=None, threshold=1E-12):
    ints_aa = np.einsum('ijkl->ljik', mohijkl)
    if mohijkl_bb is None or mohijkl_ba is None:
        ints_bb = ints_ba = ints_ab = ints_aa
    else:
        ints_bb = np.einsum('ijkl->ljik', mohijkl_bb)
        ints_ba = np.einsum('ijkl->ljik', mohijkl_ba)
        ints_ab = np.einsum('ijkl->ljik', mohijkl_ba.transpose())

    # The number of spin orbitals is twice the number of orbitals
    norbs = mohijkl.shape[0]
    nspin_orbs = 2*norbs
    moh2_qubit = np.zeros([nspin_orbs, nspin_orbs, nspin_orbs, nspin_orbs])
    for p in range(nspin_orbs):  # pylint: disable=invalid-name
        for q in range(nspin_orbs):
            for r in range(nspin_orbs):
                for s in range(nspin_orbs):  # pylint: disable=invalid-name
                    spinp = int(p/norbs)
                    spinq = int(q/norbs)
                    spinr = int(r/norbs)
                    spins = int(s/norbs)
                    if spinp != spins:
                        continue
                    if spinq != spinr:
                        continue
                    if spinp == 0:
                        ints = ints_aa if spinq == 0 else ints_ba
                    else:
                        ints = ints_ab if spinq == 0 else ints_bb
                    orbp = int(p % norbs)
                    orbq = int(q % norbs)
                    orbr = int(r % norbs)
                    orbs = int(s % norbs)
                    if abs(ints[orbp, orbq, orbr, orbs]) > threshold:
                        moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]
    return moh2_qubit

def _make_np_bool(arr):
    if not isinstance(arr, (list, np.ndarray, tuple)):
        arr = [arr]
    arr = np.asarray(arr).astype(np.bool)
    return arr

def _count_set_bits(i):
    """
    Counts the number of set bits in a uint (or a numpy array of uints).
    """
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24

class Pauli:
    """A simple class representing Pauli Operators.

    The form is P_zx = (-i)^dot(z,x) Z^z X^x where z and x are elements of Z_2^n.
    That is, there are 4^n elements (no phases in this group).

    For example, for 1 qubit
    P_00 = Z^0 X^0 = I
    P_01 = X
    P_10 = Z
    P_11 = -iZX = (-i) iY = Y

    The overload __mul__ does not track the sign: P1*P2 = Z^(z1+z2) X^(x1+x2) but
    sgn_prod does __mul__ and track the phase: P1*P2 = (-i)^dot(z1+z2,x1+x2) Z^(z1+z2) X^(x1+x2)
    where the sums are taken modulo 2.

    Pauli vectors z and x are supposed to be defined as boolean numpy arrays.

    Ref.
    Jeroen Dehaene and Bart De Moor
    Clifford group, stabilizer states, and linear and quadratic operations
    over GF(2)
    Phys. Rev. A 68, 042318 – Published 20 October 2003
    """

    def __init__(self, z=None, x=None, label=None):
        r"""Make the Pauli object.

        Note that, for the qubit index:
            - Order of z, x vectors is q_0 ... q_{n-1},
            - Order of pauli label is q_{n-1} ... q_0

        E.g.,
            - z and x vectors: z = [z_0 ... z_{n-1}], x = [x_0 ... x_{n-1}]
            - a pauli is $P_{n-1} \otimes ... \otimes P_0$

        Args:
            z (numpy.ndarray): boolean, z vector
            x (numpy.ndarray): boolean, x vector
            label (str): pauli label
        """
        if label is not None:
            a = Pauli.from_label(label)
            self._z = a.z
            self._x = a.x
        else:
            self._init_from_bool(z, x)

    @classmethod
    def from_label(cls, label):
        r"""Take pauli string to construct pauli.

        The qubit index of pauli label is q_{n-1} ... q_0.
        E.g., a pauli is $P_{n-1} \otimes ... \otimes P_0$

        Args:
            label (str): pauli label

        Returns:
            Pauli: the constructed pauli

        Raises:
            QiskitError: invalid character in the label
        """
        z = np.zeros(len(label), dtype=np.bool)
        x = np.zeros(len(label), dtype=np.bool)
        for i, char in enumerate(label):
            if char == 'X':
                x[-i - 1] = True
            elif char == 'Z':
                z[-i - 1] = True
            elif char == 'Y':
                z[-i - 1] = True
                x[-i - 1] = True
            elif char != 'I':
                raise print("Pauli string must be only consisted of 'I', 'X', "
                                  "'Y' or 'Z' but you have {}.".format(char))
        return cls(z=z, x=x)


    def _init_from_bool(self, z, x):
        """Construct pauli from boolean array.

        Args:
            z (numpy.ndarray): boolean, z vector
            x (numpy.ndarray): boolean, x vector

        Returns:
            Pauli: self

        Raises:
            QiskitError: if z or x are None or the length of z and x are different.
        """
        if z is None:
            raise print("z vector must not be None.")
        if x is None:
            raise print("x vector must not be None.")
        if len(z) != len(x):
            raise print("length of z and x vectors must be "
                              "the same. (z: {} vs x: {})".format(len(z), len(x)))

        z = _make_np_bool(z)
        x = _make_np_bool(x)
        self._z = z
        self._x = x

        return self

    def __len__(self):
        """Return number of qubits."""
        return len(self._z)


    def __repr__(self):
        """Return the representation of self."""
        z = list(self._z)
        x = list(self._x)

        ret = self.__class__.__name__ + "(z={}, x={})".format(z, x)
        return ret

    def __str__(self):
        """Output the Pauli label."""
        label = ''
        for z, x in zip(self._z[::-1], self._x[::-1]):
            if not z and not x:
                label = ''.join([label, 'I'])
            elif not z and x:
                label = ''.join([label, 'X'])
            elif z and not x:
                label = ''.join([label, 'Z'])
            else:
                label = ''.join([label, 'Y'])
        return label

    def __eq__(self, other):
        """Return True if all Pauli terms are equal.

        Args:
            other (Pauli): other pauli

        Returns:
            bool: are self and other equal.
        """
        res = False
        if len(self) == len(other):
            if np.all(self._z == other.z) and np.all(self._x == other.x):
                res = True
        return res

    def __mul__(self, other):
        """Multiply two Paulis.

        Returns:
            Pauli: the multiplied pauli.

        Raises:
            QiskitError: if the number of qubits of two paulis are different.
        """
        if len(self) != len(other):
            raise print("These Paulis cannot be multiplied - different "
                              "number of qubits. ({} vs {})".format(len(self), len(other)))
        z_new = np.logical_xor(self._z, other.z)
        x_new = np.logical_xor(self._x, other.x)
        return Pauli(z_new, x_new)


    def __imul__(self, other):
        """Multiply two Paulis.

        Returns:
            Pauli: the multiplied pauli and save to itself, in-place computation.

        Raises:
            QiskitError: if the number of qubits of two paulis are different.
        """
        if len(self) != len(other):
            raise print("These Paulis cannot be multiplied - different "
                              "number of qubits. ({} vs {})".format(len(self), len(other)))
        self._z = np.logical_xor(self._z, other.z)
        self._x = np.logical_xor(self._x, other.x)
        return self

    def __hash__(self):
        """Make object is hashable, based on the pauli label to hash."""
        return hash(str(self))

    @property
    def z(self):
        """Getter of z."""
        return self._z

    @property
    def x(self):
        """Getter of x."""
        return self._x

    @staticmethod
    def sgn_prod(p1, p2):
        r"""
        Multiply two Paulis and track the phase.

        $P_3 = P_1 \otimes P_2$: X*Y

        Args:
            p1 (Pauli): pauli 1
            p2 (Pauli): pauli 2

        Returns:
            Pauli: the multiplied pauli
            complex: the sign of the multiplication, 1, -1, 1j or -1j
        """
        phase = Pauli._prod_phase(p1, p2)
        new_pauli = p1 * p2
        return new_pauli, phase


    @property
    def num_qubits(self):
        """Number of qubits."""
        return len(self)

    def to_label(self):
        """Present the pauli labels in I, X, Y, Z format.

        Order is $q_{n-1} .... q_0$

        Returns:
            str: pauli label
        """
        return str(self)


    def to_matrix(self):
        r"""
        Convert Pauli to a matrix representation.

        Order is q_{n-1} .... q_0, i.e., $P_{n-1} \otimes ... P_0$

        Returns:
            numpy.array: a matrix that represents the pauli.
        """
        mat = self.to_spmatrix()
        return mat.toarray()


    def to_spmatrix(self):
        r"""
        Convert Pauli to a sparse matrix representation (CSR format).

        Order is q_{n-1} .... q_0, i.e., $P_{n-1} \otimes ... P_0$

        Returns:
            scipy.sparse.csr_matrix: a sparse matrix with CSR format that
            represents the pauli.
        """
        _x, _z = self._x, self._z
        n = 2**len(_x)
        twos_array = 1 << np.arange(len(_x))
        xs = np.array(_x).dot(twos_array)
        zs = np.array(_z).dot(twos_array)
        rows = np.arange(n+1, dtype=np.uint)
        columns = rows ^ xs
        global_factor = (-1j)**np.dot(np.array(_x, dtype=np.uint), _z)
        data = global_factor*(-1)**np.mod(_count_set_bits(zs & rows), 2)
        return sparse.csr_matrix((data, columns, rows), shape=(n, n))


    def to_operator(self):
        """Convert to Operator object."""
        # Place import here to avoid cyclic import from circuit visualization
        from qiskit.quantum_info.operators.operator import Operator
        return Operator(self.to_matrix())


    def to_instruction(self):
        """Convert to Pauli circuit instruction."""
        from qiskit.circuit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
        gates = {'I': IGate(), 'X': XGate(), 'Y': YGate(), 'Z': ZGate()}
        label = self.to_label()
        num_qubits = self.num_qubits
        qreg = QuantumRegister(num_qubits)
        circuit = QuantumCircuit(qreg, name='Pauli:{}'.format(label))
        for i, pauli in enumerate(reversed(label)):
            circuit.append(gates[pauli], [qreg[i]])
        return circuit.to_instruction()


    def update_z(self, z, indices=None):
        """
        Update partial or entire z.

        Args:
            z (numpy.ndarray or list): to-be-updated z
            indices (numpy.ndarray or list or optional): to-be-updated qubit indices

        Returns:
            Pauli: self

        Raises:
            QiskitError: when updating whole z, the number of qubits must be the same.
        """
        z = _make_np_bool(z)
        if indices is None:
            if len(self._z) != len(z):
                raise print("During updating whole z, you can not "
                                  "change the number of qubits.")
            self._z = z
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self._z[idx] = z[p]

        return self


    def update_x(self, x, indices=None):
        """
        Update partial or entire x.

        Args:
            x (numpy.ndarray or list): to-be-updated x
            indices (numpy.ndarray or list or optional): to-be-updated qubit indices

        Returns:
            Pauli: self

        Raises:
            QiskitError: when updating whole x, the number of qubits must be the same.
        """
        x = _make_np_bool(x)
        if indices is None:
            if len(self._x) != len(x):
                raise print("During updating whole x, you can not change "
                                  "the number of qubits.")
            self._x = x
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self._x[idx] = x[p]

        return self


    def insert_paulis(self, indices=None, paulis=None, pauli_labels=None):
        """
        Insert or append pauli to the targeted indices.

        If indices is None, it means append at the end.

        Args:
            indices (list[int]): the qubit indices to be inserted
            paulis (Pauli): the to-be-inserted or appended pauli
            pauli_labels (list[str]): the to-be-inserted or appended pauli label

        Note:
            the indices refers to the location of original paulis,
            e.g. if indices = [0, 2], pauli_labels = ['Z', 'I'] and original pauli = 'ZYXI'
            the pauli will be updated to ZY'I'XI'Z'
            'Z' and 'I' are inserted before the qubit at 0 and 2.

        Returns:
            Pauli: self

        Raises:
            QiskitError: provide both `paulis` and `pauli_labels` at the same time
        """
        if pauli_labels is not None:
            if paulis is not None:
                raise print("Please only provide either `paulis` or `pauli_labels`")
            if isinstance(pauli_labels, str):
                pauli_labels = list(pauli_labels)
            # since pauli label is in reversed order.
            paulis = Pauli.from_label(pauli_labels[::-1])

        if indices is None:  # append
            self._z = np.concatenate((self._z, paulis.z))
            self._x = np.concatenate((self._x, paulis.x))
        else:
            if not isinstance(indices, list):
                indices = [indices]
            self._z = np.insert(self._z, indices, paulis.z)
            self._x = np.insert(self._x, indices, paulis.x)

        return self

def BK(self, n):
        """
        Bravyi-Kitaev mode.

        Args:
            n (int): number of modes

         Returns:
             numpy.ndarray: Array of mode indexes
        """
        def parity_set(j, n):
            """
            Computes the parity set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes

            if j < n / 2:
                indexes = np.append(indexes, parity_set(j, n / 2))
            else:
                indexes = np.append(indexes, np.append(
                    parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1))
            return indexes

        def update_set(j, n):
            """
            Computes the update set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes
            if j < n / 2:
                indexes = np.append(indexes, np.append(
                    n - 1, update_set(j, n / 2)))
            else:
                indexes = np.append(indexes, update_set(j - n / 2, n / 2) + n / 2)
            return indexes

        def flip_set(j, n):
            """
            Computes the flip set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes
            if j < n / 2:
                indexes = np.append(indexes, flip_set(j, n / 2))
            elif j >= n / 2 and j < n - 1:  # pylint: disable=chained-comparison
                indexes = np.append(indexes, flip_set(j - n / 2, n / 2) + n / 2)
            else:
                indexes = np.append(np.append(indexes, flip_set(
                    j - n / 2, n / 2) + n / 2), n / 2 - 1)
            return indexes

        a_list = []
        # FIND BINARY SUPERSET SIZE
        bin_sup = 1
        # pylint: disable=comparison-with-callable
        while n > np.power(2, bin_sup):
            bin_sup += 1
        # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
        update_sets = []
        update_pauli = []

        parity_sets = []
        parity_pauli = []

        flip_sets = []

        remainder_sets = []
        remainder_pauli = []
        for j in range(n):

            update_sets.append(update_set(j, np.power(2, bin_sup)))
            update_sets[j] = update_sets[j][update_sets[j] < n]

            parity_sets.append(parity_set(j, np.power(2, bin_sup)))
            parity_sets[j] = parity_sets[j][parity_sets[j] < n]

            flip_sets.append(flip_set(j, np.power(2, bin_sup)))
            flip_sets[j] = flip_sets[j][flip_sets[j] < n]

            remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

            update_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            parity_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            remainder_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            for k in range(n):
                if np.in1d(k, update_sets[j]):
                    update_pauli[j].update_x(True, k)
                if np.in1d(k, parity_sets[j]):
                    parity_pauli[j].update_z(True, k)
                if np.in1d(k, remainder_sets[j]):
                    remainder_pauli[j].update_z(True, k)

            x_j = Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool))
            x_j.update_x(True, j)
            y_j = Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool))
            y_j.update_z(True, j)
            y_j.update_x(True, j)
            a_list.append((update_pauli[j] * x_j * parity_pauli[j],
                           update_pauli[j] * y_j * remainder_pauli[j]))
            return a_list



# Test region
h3 = molecule([
    (1,0,0,0),
    (1,0,0,1.3),
    (1,0,0,2.6)],
    units='angs',spin=1)

bfs = basisset(h3,'sto3g')

i1 = onee_integrals(bfs,h3)
i2 = twoe_integrals(bfs)
S=i1.S
T=i1.T
V=i1.V
solver = uhf(h3, bfs)
ehf = solver.converge()
hij=T+V
hijkl=i2
if hasattr(solver, 'orbs'):
    orbs = solver.orbs
    orbs_b = None
else:
    orbs = solver.orbsa
    orbs_b = solver.orbsb
norbs = len(orbs)
if hasattr(solver, 'orbe'):
    orbs_energy = solver.orbe
    orbs_energy_b = None
else:
    orbs_energy = solver.orbea
    orbs_energy_b = solver.orbeb
enuke = h3.nuclear_repulsion()

mohij = simx(hij, orbs)
mohij_b = None
if orbs_b is not None:
    mohij_b = simx(hij, orbs_b)
    
    
mohijkl = hijkl.transform(orbs)  
mohijkl_bb = None
mohijkl_ba = None
if orbs_b is not None:
    mohijkl_bb = hijkl.transform(orbs_b)
    mohijkl_ba = np.einsum('aI,bJ,cK,dL,abcd->IJKL', orbs_b, orbs_b, orbs, orbs, hijkl[...])

    
hpq=onee_to_spin(mohij,mohij_b,threshold=1e-12)
hpqrs=twoe_to_spin(mohijkl,mohijkl_bb,mohijkl_ba,threshold=1e-12)    
    
    
    
    
    
    
    
    
    
