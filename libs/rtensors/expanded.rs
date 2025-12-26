pub mod protocol {
    use serde::{Deserialize, Serialize};
    use crate::{
        backend::remote::client::RemoteBuf,
        core::{
            meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError,
            value::{DType, TensorValue},
            MetaTensor,
        },
        ops::base::BinaryOpType,
    };
    pub(crate) struct Slice {
        pub(crate) data: Vec<u8>,
        pub(crate) dtype: DType,
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for Slice {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private228::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "Slice",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "data",
                    &self.data,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "dtype",
                    &self.dtype,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for Slice {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private228::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private228::Ok(__Field::__field0),
                            1u64 => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "data" => _serde::__private228::Ok(__Field::__field0),
                            "dtype" => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"data" => _serde::__private228::Ok(__Field::__field0),
                            b"dtype" => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private228::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de> {
                    marker: _serde::__private228::PhantomData<Slice>,
                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = Slice;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "struct Slice",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            Vec<u8>,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct Slice with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            DType,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct Slice with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private228::Ok(Slice {
                            data: __field0,
                            dtype: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private228::Option<Vec<u8>> = _serde::__private228::None;
                        let mut __field1: _serde::__private228::Option<DType> = _serde::__private228::None;
                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                            __Field,
                        >(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private228::Option::is_some(&__field0) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("data"),
                                        );
                                    }
                                    __field0 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<Vec<u8>>(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private228::Option::is_some(&__field1) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("dtype"),
                                        );
                                    }
                                    __field1 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<DType>(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private228::Some(__field0) => __field0,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("data")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private228::Some(__field1) => __field1,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("dtype")?
                            }
                        };
                        _serde::__private228::Ok(Slice {
                            data: __field0,
                            dtype: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["data", "dtype"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "Slice",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private228::PhantomData::<Slice>,
                        lifetime: _serde::__private228::PhantomData,
                    },
                )
            }
        }
    };
    impl<T: TensorValue> From<Slice> for Result<Box<[T]>, TensorError> {
        fn from(val: Slice) -> Self {
            val.to_boxed_slice::<T>()
        }
    }
    impl Slice {
        #[inline(always)]
        pub(crate) fn from_boxed_slice<T: TensorValue>(boxed: Box<[T]>) -> Self {
            let dtype = T::DTYPE;
            let data = unsafe {
                let len = boxed.len() * std::mem::size_of::<T>();
                let ptr = Box::into_raw(boxed) as *mut u8;
                Vec::from_raw_parts(ptr, len, len)
            };
            Self { data, dtype }
        }
        #[inline(always)]
        pub(crate) fn from_slice<T: TensorValue>(slice: &[T]) -> Self {
            let dtype = T::DTYPE;
            let data = unsafe {
                let len = std::mem::size_of_val(slice);
                let ptr = slice.as_ptr() as *const u8;
                let mut vec = Vec::with_capacity(len);
                vec.set_len(len);
                std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), len);
                vec
            };
            Self { data, dtype }
        }
        #[inline(always)]
        pub(crate) fn to_boxed_slice<T: TensorValue>(
            self,
        ) -> Result<Box<[T]>, TensorError> {
            if self.dtype != T::DTYPE {
                return Err(
                    TensorError::BackendError(
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "Type mismatch: expected {0:?}, got {1:?}",
                                    T::DTYPE,
                                    self.dtype,
                                ),
                            )
                        }),
                    ),
                );
            }
            let boxed = unsafe {
                let len = self.data.len() / std::mem::size_of::<T>();
                let ptr = self.data.as_ptr() as *mut T;
                std::mem::forget(self.data);
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len))
            };
            Ok(boxed)
        }
    }
    impl<T: TensorValue> From<Box<[T]>> for Slice {
        fn from(boxed: Box<[T]>) -> Self {
            Slice::from_boxed_slice(boxed)
        }
    }
    impl<T: TensorValue> From<&[T]> for Slice {
        fn from(slice: &[T]) -> Self {
            Slice::from_slice(slice)
        }
    }
    pub(crate) struct Value {
        data: Vec<u8>,
        dtype: DType,
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for Value {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private228::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "Value",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "data",
                    &self.data,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "dtype",
                    &self.dtype,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for Value {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private228::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private228::Ok(__Field::__field0),
                            1u64 => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "data" => _serde::__private228::Ok(__Field::__field0),
                            "dtype" => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"data" => _serde::__private228::Ok(__Field::__field0),
                            b"dtype" => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private228::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de> {
                    marker: _serde::__private228::PhantomData<Value>,
                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = Value;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "struct Value",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            Vec<u8>,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct Value with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            DType,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct Value with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private228::Ok(Value {
                            data: __field0,
                            dtype: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private228::Option<Vec<u8>> = _serde::__private228::None;
                        let mut __field1: _serde::__private228::Option<DType> = _serde::__private228::None;
                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                            __Field,
                        >(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private228::Option::is_some(&__field0) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("data"),
                                        );
                                    }
                                    __field0 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<Vec<u8>>(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private228::Option::is_some(&__field1) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("dtype"),
                                        );
                                    }
                                    __field1 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<DType>(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private228::Some(__field0) => __field0,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("data")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private228::Some(__field1) => __field1,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("dtype")?
                            }
                        };
                        _serde::__private228::Ok(Value {
                            data: __field0,
                            dtype: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["data", "dtype"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "Value",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private228::PhantomData::<Value>,
                        lifetime: _serde::__private228::PhantomData,
                    },
                )
            }
        }
    };
    impl<T: TensorValue> From<Value> for Result<T, TensorError> {
        fn from(val: Value) -> Self {
            val.to_value::<T>()
        }
    }
    impl Value {
        #[inline(always)]
        pub(crate) fn from_value<T: TensorValue>(value: T) -> Self {
            let dtype = T::DTYPE;
            let data = unsafe {
                let size = std::mem::size_of::<T>();
                let ptr = &value as *const T as *const u8;
                let mut vec = Vec::with_capacity(size);
                vec.set_len(size);
                std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), size);
                vec
            };
            Self { data, dtype }
        }
        #[inline(always)]
        pub(crate) fn to_value<T: TensorValue>(self) -> Result<T, TensorError> {
            if self.dtype != T::DTYPE {
                return Err(
                    TensorError::BackendError(
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "Type mismatch: expected {0:?}, got {1:?}",
                                    T::DTYPE,
                                    self.dtype,
                                ),
                            )
                        }),
                    ),
                );
            }
            let value = unsafe {
                let ptr = self.data.as_ptr() as *const T;
                std::ptr::read(ptr)
            };
            Ok(value)
        }
    }
    impl<T: TensorValue> From<T> for Value {
        fn from(value: T) -> Self {
            Value::from_value(value)
        }
    }
    pub(crate) struct TypelessBuf {
        pub(crate) id: u32,
        pub(crate) dtype: DType,
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for TypelessBuf {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private228::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "TypelessBuf",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "id",
                    &self.id,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "dtype",
                    &self.dtype,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for TypelessBuf {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private228::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private228::Ok(__Field::__field0),
                            1u64 => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "id" => _serde::__private228::Ok(__Field::__field0),
                            "dtype" => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"id" => _serde::__private228::Ok(__Field::__field0),
                            b"dtype" => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private228::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de> {
                    marker: _serde::__private228::PhantomData<TypelessBuf>,
                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = TypelessBuf;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "struct TypelessBuf",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            u32,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct TypelessBuf with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            DType,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct TypelessBuf with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private228::Ok(TypelessBuf {
                            id: __field0,
                            dtype: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private228::Option<u32> = _serde::__private228::None;
                        let mut __field1: _serde::__private228::Option<DType> = _serde::__private228::None;
                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                            __Field,
                        >(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private228::Option::is_some(&__field0) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("id"),
                                        );
                                    }
                                    __field0 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<u32>(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private228::Option::is_some(&__field1) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("dtype"),
                                        );
                                    }
                                    __field1 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<DType>(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private228::Some(__field0) => __field0,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("id")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private228::Some(__field1) => __field1,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("dtype")?
                            }
                        };
                        _serde::__private228::Ok(TypelessBuf {
                            id: __field0,
                            dtype: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["id", "dtype"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "TypelessBuf",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private228::PhantomData::<TypelessBuf>,
                        lifetime: _serde::__private228::PhantomData,
                    },
                )
            }
        }
    };
    #[automatically_derived]
    impl ::core::clone::Clone for TypelessBuf {
        #[inline]
        fn clone(&self) -> TypelessBuf {
            let _: ::core::clone::AssertParamIsClone<u32>;
            let _: ::core::clone::AssertParamIsClone<DType>;
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for TypelessBuf {}
    impl<T: TensorValue> From<TypelessBuf> for Result<RemoteBuf<T>, TensorError> {
        fn from(val: TypelessBuf) -> Self {
            Ok(RemoteBuf::from_typeless(val))
        }
    }
    pub(crate) struct Response {
        pub(crate) asynchronous: bool,
        pub(crate) complete: bool,
        pub(crate) task_id: u32,
        pub(crate) message: Messages,
        pub(crate) error: Option<TensorError>,
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for Response {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private228::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "Response",
                    false as usize + 1 + 1 + 1 + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "asynchronous",
                    &self.asynchronous,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "complete",
                    &self.complete,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "task_id",
                    &self.task_id,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "message",
                    &self.message,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "error",
                    &self.error,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for Response {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private228::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __field2,
                    __field3,
                    __field4,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private228::Ok(__Field::__field0),
                            1u64 => _serde::__private228::Ok(__Field::__field1),
                            2u64 => _serde::__private228::Ok(__Field::__field2),
                            3u64 => _serde::__private228::Ok(__Field::__field3),
                            4u64 => _serde::__private228::Ok(__Field::__field4),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "asynchronous" => _serde::__private228::Ok(__Field::__field0),
                            "complete" => _serde::__private228::Ok(__Field::__field1),
                            "task_id" => _serde::__private228::Ok(__Field::__field2),
                            "message" => _serde::__private228::Ok(__Field::__field3),
                            "error" => _serde::__private228::Ok(__Field::__field4),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"asynchronous" => {
                                _serde::__private228::Ok(__Field::__field0)
                            }
                            b"complete" => _serde::__private228::Ok(__Field::__field1),
                            b"task_id" => _serde::__private228::Ok(__Field::__field2),
                            b"message" => _serde::__private228::Ok(__Field::__field3),
                            b"error" => _serde::__private228::Ok(__Field::__field4),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private228::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de> {
                    marker: _serde::__private228::PhantomData<Response>,
                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = Response;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "struct Response",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            bool,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct Response with 5 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            bool,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct Response with 5 elements",
                                    ),
                                );
                            }
                        };
                        let __field2 = match _serde::de::SeqAccess::next_element::<
                            u32,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        2usize,
                                        &"struct Response with 5 elements",
                                    ),
                                );
                            }
                        };
                        let __field3 = match _serde::de::SeqAccess::next_element::<
                            Messages,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        3usize,
                                        &"struct Response with 5 elements",
                                    ),
                                );
                            }
                        };
                        let __field4 = match _serde::de::SeqAccess::next_element::<
                            Option<TensorError>,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        4usize,
                                        &"struct Response with 5 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private228::Ok(Response {
                            asynchronous: __field0,
                            complete: __field1,
                            task_id: __field2,
                            message: __field3,
                            error: __field4,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private228::Option<bool> = _serde::__private228::None;
                        let mut __field1: _serde::__private228::Option<bool> = _serde::__private228::None;
                        let mut __field2: _serde::__private228::Option<u32> = _serde::__private228::None;
                        let mut __field3: _serde::__private228::Option<Messages> = _serde::__private228::None;
                        let mut __field4: _serde::__private228::Option<
                            Option<TensorError>,
                        > = _serde::__private228::None;
                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                            __Field,
                        >(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private228::Option::is_some(&__field0) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "asynchronous",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private228::Option::is_some(&__field1) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "complete",
                                            ),
                                        );
                                    }
                                    __field1 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                    );
                                }
                                __Field::__field2 => {
                                    if _serde::__private228::Option::is_some(&__field2) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "task_id",
                                            ),
                                        );
                                    }
                                    __field2 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<u32>(&mut __map)?,
                                    );
                                }
                                __Field::__field3 => {
                                    if _serde::__private228::Option::is_some(&__field3) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "message",
                                            ),
                                        );
                                    }
                                    __field3 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<Messages>(&mut __map)?,
                                    );
                                }
                                __Field::__field4 => {
                                    if _serde::__private228::Option::is_some(&__field4) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("error"),
                                        );
                                    }
                                    __field4 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<
                                            Option<TensorError>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private228::Some(__field0) => __field0,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("asynchronous")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private228::Some(__field1) => __field1,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("complete")?
                            }
                        };
                        let __field2 = match __field2 {
                            _serde::__private228::Some(__field2) => __field2,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("task_id")?
                            }
                        };
                        let __field3 = match __field3 {
                            _serde::__private228::Some(__field3) => __field3,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("message")?
                            }
                        };
                        let __field4 = match __field4 {
                            _serde::__private228::Some(__field4) => __field4,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("error")?
                            }
                        };
                        _serde::__private228::Ok(Response {
                            asynchronous: __field0,
                            complete: __field1,
                            task_id: __field2,
                            message: __field3,
                            error: __field4,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &[
                    "asynchronous",
                    "complete",
                    "task_id",
                    "message",
                    "error",
                ];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "Response",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private228::PhantomData::<Response>,
                        lifetime: _serde::__private228::PhantomData,
                    },
                )
            }
        }
    };
    pub(crate) struct Request {
        pub(crate) task_id: u32,
        pub(crate) message: Messages,
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for Request {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private228::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "Request",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "task_id",
                    &self.task_id,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "message",
                    &self.message,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for Request {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private228::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private228::Ok(__Field::__field0),
                            1u64 => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "task_id" => _serde::__private228::Ok(__Field::__field0),
                            "message" => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"task_id" => _serde::__private228::Ok(__Field::__field0),
                            b"message" => _serde::__private228::Ok(__Field::__field1),
                            _ => _serde::__private228::Ok(__Field::__ignore),
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private228::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de> {
                    marker: _serde::__private228::PhantomData<Request>,
                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = Request;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "struct Request",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            u32,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct Request with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            Messages,
                        >(&mut __seq)? {
                            _serde::__private228::Some(__value) => __value,
                            _serde::__private228::None => {
                                return _serde::__private228::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct Request with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private228::Ok(Request {
                            task_id: __field0,
                            message: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private228::Option<u32> = _serde::__private228::None;
                        let mut __field1: _serde::__private228::Option<Messages> = _serde::__private228::None;
                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                            __Field,
                        >(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private228::Option::is_some(&__field0) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "task_id",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<u32>(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private228::Option::is_some(&__field1) {
                                        return _serde::__private228::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "message",
                                            ),
                                        );
                                    }
                                    __field1 = _serde::__private228::Some(
                                        _serde::de::MapAccess::next_value::<Messages>(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private228::Some(__field0) => __field0,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("task_id")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private228::Some(__field1) => __field1,
                            _serde::__private228::None => {
                                _serde::__private228::de::missing_field("message")?
                            }
                        };
                        _serde::__private228::Ok(Request {
                            task_id: __field0,
                            message: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["task_id", "message"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "Request",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private228::PhantomData::<Request>,
                        lifetime: _serde::__private228::PhantomData,
                    },
                )
            }
        }
    };
    impl Request {
        #[inline(always)]
        pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
            if true {
                if !!self.message.is_response() {
                    ::core::panicking::panic(
                        "assertion failed: !self.message.is_response()",
                    )
                }
            }
            bincode::serialize(self)
        }
        #[inline(always)]
        pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
            let resp: Request = bincode::deserialize(data)?;
            if true {
                if !!resp.message.is_response() {
                    ::core::panicking::panic(
                        "assertion failed: !resp.message.is_response()",
                    )
                }
            }
            Ok(resp)
        }
    }
    impl Response {
        #[inline(always)]
        pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
            if true {
                if !self.message.is_response() {
                    ::core::panicking::panic(
                        "assertion failed: self.message.is_response()",
                    )
                }
            }
            bincode::serialize(self)
        }
        #[inline(always)]
        pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
            let resp: Response = bincode::deserialize(data)?;
            if true {
                if !resp.message.is_response() {
                    ::core::panicking::panic(
                        "assertion failed: resp.message.is_response()",
                    )
                }
            }
            Ok(resp)
        }
    }
    impl<T: TensorValue> From<RemoteBuf<T>> for TypelessBuf {
        fn from(buf: RemoteBuf<T>) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<f32> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<f32>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<f64> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<f64>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<i8> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<i8>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<i16> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<i16>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<i32> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<i32>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<i64> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<i64>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<i128> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<i128>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<u8> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<u8>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<u16> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<u16>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<u32> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<u32>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<u64> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<u64>,
            }
        }
    }
    impl From<TypelessBuf> for RemoteBuf<u128> {
        fn from(buf: TypelessBuf) -> Self {
            Self {
                id: buf.id,
                dtype: buf.dtype,
                _marker: std::marker::PhantomData::<u128>,
            }
        }
    }
    pub(crate) enum Messages {
        ErrorResponse { message: String },
        DeviceType,
        DeviceTypeResponse { device_type: DeviceType },
        AllocFromSlice { src: Slice },
        AllocFromSliceResponse(Result<TypelessBuf, TensorError>),
        Alloc { len: usize, dtype: DType },
        AllocResponse(Result<TypelessBuf, TensorError>),
        CopyFromSlice { dst: TypelessBuf, src: Slice },
        CopyFromSliceResponse(Result<(), TensorError>),
        Read { buf: TypelessBuf, offset: usize },
        ReadResponse(Result<Value, TensorError>),
        Write { buf: TypelessBuf, offset: usize, value: Value },
        WriteResponse(Result<(), TensorError>),
        Len { buf: TypelessBuf },
        LenResponse(usize),
        Copy { src: TypelessBuf },
        CopyResponse(Result<TypelessBuf, TensorError>),
        Dump { src: TypelessBuf },
        DumpResponse(Result<Slice, TensorError>),
        ApplyElementwiseBinary1dStrided {
            buf: TypelessBuf,
            op: (BinaryOpType, Value),
            offset: usize,
            stride: isize,
            len: usize,
        },
        ApplyElementwiseBinary1dStridedResponse(Result<(), TensorError>),
        ApplyElementwiseBinaryContiguous {
            buf: TypelessBuf,
            op: (BinaryOpType, Value),
            start: usize,
            len: usize,
        },
        ApplyElementwiseBinaryContiguousResponse(Result<(), TensorError>),
        ApplyElementwiseBinaryNd {
            buf: TypelessBuf,
            op: (BinaryOpType, Value),
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplyElementwiseBinaryNdResponse(Result<(), TensorError>),
        Broadcast {
            left: (TypelessBuf, MetaTensor),
            right: (TypelessBuf, MetaTensor),
            dst: (TypelessBuf, MetaTensor),
            op: BinaryOpType,
        },
        BroadcastResponse(Result<(), TensorError>),
        ApplyNegContiguous { buf: TypelessBuf, start: usize, len: usize },
        ApplyNegContiguousResponse(Result<(), TensorError>),
        ApplyNeg1dStrided { buf: TypelessBuf, offset: usize, stride: isize, len: usize },
        ApplyNeg1dStridedResponse(Result<(), TensorError>),
        ApplyNegNd {
            buf: TypelessBuf,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplyNegNdResponse(Result<(), TensorError>),
        Matmul {
            lhs: (TypelessBuf, MetaTensor),
            rhs: (TypelessBuf, MetaTensor),
            dst: TypelessBuf,
            b: usize,
            m: usize,
            k: usize,
            n: usize,
            contiguity: ContiguityTypes,
        },
        MatmulResponse(Result<(), TensorError>),
        ApplyReluNd {
            buf: TypelessBuf,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplyReluNdResponse(Result<(), TensorError>),
        ApplyRelu1dStrided {
            buf: TypelessBuf,
            offset: usize,
            stride: isize,
            len: usize,
        },
        ApplyRelu1dStridedResponse(Result<(), TensorError>),
        ApplyReluContiguous { buf: TypelessBuf, offset: usize, len: usize },
        ApplyReluContiguousResponse(Result<(), TensorError>),
        ApplySigmoidNd {
            buf: TypelessBuf,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplySigmoidNdResponse(Result<(), TensorError>),
        ApplySigmoid1dStrided {
            buf: TypelessBuf,
            offset: usize,
            stride: isize,
            len: usize,
        },
        ApplySigmoid1dStridedResponse(Result<(), TensorError>),
        ApplySigmoidContiguous { buf: TypelessBuf, offset: usize, len: usize },
        ApplySigmoidContiguousResponse(Result<(), TensorError>),
        ApplyTanhNd {
            buf: TypelessBuf,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<isize>,
        },
        ApplyTanhNdResponse(Result<(), TensorError>),
        ApplyTanh1dStrided {
            buf: TypelessBuf,
            offset: usize,
            stride: isize,
            len: usize,
        },
        ApplyTanh1dStridedResponse(Result<(), TensorError>),
        ApplyTanhContiguous { buf: TypelessBuf, offset: usize, len: usize },
        ApplyTanhContiguousResponse(Result<(), TensorError>),
        CopyRangeWithin {
            dst: TypelessBuf,
            src: TypelessBuf,
            dst_offset: usize,
            src_offset: usize,
            len: usize,
        },
        CopyRangeWithinResponse(Result<(), TensorError>),
        ActionCompleted(u32),
    }
    fn handle_request_(message: Messages, task_id: u32, connection: Temp) {
        match message {
            Messages::Alloc { len, dtype } => {
                let result = super::enumdispatch::dispatch_alloc(len, dtype, connection);
                let message = Messages::AllocResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::AllocFromSlice { src } => {
                let result = super::enumdispatch::dispatch_alloc_from_slice(
                    src,
                    connection,
                );
                let message = Messages::AllocFromSliceResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyElementwiseBinaryNd { buf, op, offset, shape, stride } => {
                let result = super::enumdispatch::dispatch_apply_elementwise_binary_nd(
                    buf,
                    op,
                    offset,
                    shape,
                    stride,
                    connection,
                );
                let message = Messages::ApplyElementwiseBinaryNdResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyTanh1dStrided { buf, offset, stride, len } => {
                let result = super::enumdispatch::dispatch_apply_tanh_1d_strided(
                    buf,
                    offset,
                    stride,
                    len,
                    connection,
                );
                let message = Messages::ApplyTanh1dStridedResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::Read { buf, offset } => {
                let result = super::enumdispatch::dispatch_read(buf, offset, connection);
                let message = Messages::ReadResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyElementwiseBinaryContiguous { buf, op, start, len } => {
                let result = super::enumdispatch::dispatch_apply_elementwise_binary_contiguous(
                    buf,
                    op,
                    start,
                    len,
                    connection,
                );
                let message = Messages::ApplyElementwiseBinaryContiguousResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::Copy { src } => {
                let result = super::enumdispatch::dispatch_copy(src, connection);
                let message = Messages::CopyResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplySigmoidNd { buf, offset, shape, stride } => {
                let result = super::enumdispatch::dispatch_apply_sigmoid_nd(
                    buf,
                    offset,
                    shape,
                    stride,
                    connection,
                );
                let message = Messages::ApplySigmoidNdResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyTanhNd { buf, offset, shape, stride } => {
                let result = super::enumdispatch::dispatch_apply_tanh_nd(
                    buf,
                    offset,
                    shape,
                    stride,
                    connection,
                );
                let message = Messages::ApplyTanhNdResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyElementwiseBinary1dStrided {
                buf,
                op,
                offset,
                stride,
                len,
            } => {
                let result = super::enumdispatch::dispatch_apply_elementwise_binary_1d_strided(
                    buf,
                    op,
                    offset,
                    stride,
                    len,
                    connection,
                );
                let message = Messages::ApplyElementwiseBinary1dStridedResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::CopyFromSlice { dst, src } => {
                let result = super::enumdispatch::dispatch_copy_from_slice(
                    dst,
                    src,
                    connection,
                );
                let message = Messages::CopyFromSliceResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyNegContiguous { buf, start, len } => {
                let result = super::enumdispatch::dispatch_apply_neg_contiguous(
                    buf,
                    start,
                    len,
                    connection,
                );
                let message = Messages::ApplyNegContiguousResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyNegNd { buf, offset, shape, stride } => {
                let result = super::enumdispatch::dispatch_apply_neg_nd(
                    buf,
                    offset,
                    shape,
                    stride,
                    connection,
                );
                let message = Messages::ApplyNegNdResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyNeg1dStrided { buf, offset, stride, len } => {
                let result = super::enumdispatch::dispatch_apply_neg_1d_strided(
                    buf,
                    offset,
                    stride,
                    len,
                    connection,
                );
                let message = Messages::ApplyNeg1dStridedResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::Matmul { lhs, rhs, dst, b, m, k, n, contiguity } => {
                let result = super::enumdispatch::dispatch_matmul(
                    lhs,
                    rhs,
                    dst,
                    b,
                    m,
                    k,
                    n,
                    contiguity,
                    connection,
                );
                let message = Messages::MatmulResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyRelu1dStrided { buf, offset, stride, len } => {
                let result = super::enumdispatch::dispatch_apply_relu_1d_strided(
                    buf,
                    offset,
                    stride,
                    len,
                    connection,
                );
                let message = Messages::ApplyRelu1dStridedResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplySigmoidContiguous { buf, offset, len } => {
                let result = super::enumdispatch::dispatch_apply_sigmoid_contiguous(
                    buf,
                    offset,
                    len,
                    connection,
                );
                let message = Messages::ApplySigmoidContiguousResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyTanhContiguous { buf, offset, len } => {
                let result = super::enumdispatch::dispatch_apply_tanh_contiguous(
                    buf,
                    offset,
                    len,
                    connection,
                );
                let message = Messages::ApplyTanhContiguousResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyReluNd { buf, offset, shape, stride } => {
                let result = super::enumdispatch::dispatch_apply_relu_nd(
                    buf,
                    offset,
                    shape,
                    stride,
                    connection,
                );
                let message = Messages::ApplyReluNdResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplyReluContiguous { buf, offset, len } => {
                let result = super::enumdispatch::dispatch_apply_relu_contiguous(
                    buf,
                    offset,
                    len,
                    connection,
                );
                let message = Messages::ApplyReluContiguousResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::ApplySigmoid1dStrided { buf, offset, stride, len } => {
                let result = super::enumdispatch::dispatch_apply_sigmoid_1d_strided(
                    buf,
                    offset,
                    stride,
                    len,
                    connection,
                );
                let message = Messages::ApplySigmoid1dStridedResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::CopyRangeWithin { dst, src, dst_offset, src_offset, len } => {
                let result = super::enumdispatch::dispatch_copy_range_within(
                    dst,
                    src,
                    dst_offset,
                    src_offset,
                    len,
                    connection,
                );
                let message = Messages::CopyRangeWithinResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::Dump { src } => {
                let result = super::enumdispatch::dispatch_dump(src, connection);
                let message = Messages::DumpResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::Write { buf, offset, value } => {
                let result = super::enumdispatch::dispatch_write(
                    buf,
                    offset,
                    value,
                    connection,
                );
                let message = Messages::WriteResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
            Messages::Broadcast { left, right, dst, op } => {
                let result = super::enumdispatch::dispatch_broadcast(
                    left,
                    right,
                    dst,
                    op,
                    connection,
                );
                let message = Messages::BroadcastResponse(result);
                let err = result.as_ref().err().cloned();
                let response = Response {
                    asynchronous: false,
                    complete: true,
                    task_id: task_id,
                    error: err,
                    message: message,
                };
                connection.queue_response(response).expect("Failed to send message");
            }
        };
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for Messages {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private228::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                match *self {
                    Messages::ErrorResponse { ref message } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            0u32,
                            "ErrorResponse",
                            0 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "message",
                            message,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::DeviceType => {
                        _serde::Serializer::serialize_unit_variant(
                            __serializer,
                            "Messages",
                            1u32,
                            "DeviceType",
                        )
                    }
                    Messages::DeviceTypeResponse { ref device_type } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            2u32,
                            "DeviceTypeResponse",
                            0 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "device_type",
                            device_type,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::AllocFromSlice { ref src } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            3u32,
                            "AllocFromSlice",
                            0 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "src",
                            src,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::AllocFromSliceResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            4u32,
                            "AllocFromSliceResponse",
                            __field0,
                        )
                    }
                    Messages::Alloc { ref len, ref dtype } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            5u32,
                            "Alloc",
                            0 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "dtype",
                            dtype,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::AllocResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            6u32,
                            "AllocResponse",
                            __field0,
                        )
                    }
                    Messages::CopyFromSlice { ref dst, ref src } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            7u32,
                            "CopyFromSlice",
                            0 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "dst",
                            dst,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "src",
                            src,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::CopyFromSliceResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            8u32,
                            "CopyFromSliceResponse",
                            __field0,
                        )
                    }
                    Messages::Read { ref buf, ref offset } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            9u32,
                            "Read",
                            0 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ReadResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            10u32,
                            "ReadResponse",
                            __field0,
                        )
                    }
                    Messages::Write { ref buf, ref offset, ref value } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            11u32,
                            "Write",
                            0 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "value",
                            value,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::WriteResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            12u32,
                            "WriteResponse",
                            __field0,
                        )
                    }
                    Messages::Len { ref buf } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            13u32,
                            "Len",
                            0 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::LenResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            14u32,
                            "LenResponse",
                            __field0,
                        )
                    }
                    Messages::Copy { ref src } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            15u32,
                            "Copy",
                            0 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "src",
                            src,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::CopyResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            16u32,
                            "CopyResponse",
                            __field0,
                        )
                    }
                    Messages::Dump { ref src } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            17u32,
                            "Dump",
                            0 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "src",
                            src,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::DumpResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            18u32,
                            "DumpResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyElementwiseBinary1dStrided {
                        ref buf,
                        ref op,
                        ref offset,
                        ref stride,
                        ref len,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            19u32,
                            "ApplyElementwiseBinary1dStrided",
                            0 + 1 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "op",
                            op,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyElementwiseBinary1dStridedResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            20u32,
                            "ApplyElementwiseBinary1dStridedResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyElementwiseBinaryContiguous {
                        ref buf,
                        ref op,
                        ref start,
                        ref len,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            21u32,
                            "ApplyElementwiseBinaryContiguous",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "op",
                            op,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "start",
                            start,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyElementwiseBinaryContiguousResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            22u32,
                            "ApplyElementwiseBinaryContiguousResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyElementwiseBinaryNd {
                        ref buf,
                        ref op,
                        ref offset,
                        ref shape,
                        ref stride,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            23u32,
                            "ApplyElementwiseBinaryNd",
                            0 + 1 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "op",
                            op,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "shape",
                            shape,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyElementwiseBinaryNdResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            24u32,
                            "ApplyElementwiseBinaryNdResponse",
                            __field0,
                        )
                    }
                    Messages::Broadcast { ref left, ref right, ref dst, ref op } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            25u32,
                            "Broadcast",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "left",
                            left,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "right",
                            right,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "dst",
                            dst,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "op",
                            op,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::BroadcastResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            26u32,
                            "BroadcastResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyNegContiguous { ref buf, ref start, ref len } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            27u32,
                            "ApplyNegContiguous",
                            0 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "start",
                            start,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyNegContiguousResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            28u32,
                            "ApplyNegContiguousResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyNeg1dStrided {
                        ref buf,
                        ref offset,
                        ref stride,
                        ref len,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            29u32,
                            "ApplyNeg1dStrided",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyNeg1dStridedResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            30u32,
                            "ApplyNeg1dStridedResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyNegNd {
                        ref buf,
                        ref offset,
                        ref shape,
                        ref stride,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            31u32,
                            "ApplyNegNd",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "shape",
                            shape,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyNegNdResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            32u32,
                            "ApplyNegNdResponse",
                            __field0,
                        )
                    }
                    Messages::Matmul {
                        ref lhs,
                        ref rhs,
                        ref dst,
                        ref b,
                        ref m,
                        ref k,
                        ref n,
                        ref contiguity,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            33u32,
                            "Matmul",
                            0 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "lhs",
                            lhs,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "rhs",
                            rhs,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "dst",
                            dst,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "b",
                            b,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "m",
                            m,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "k",
                            k,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "n",
                            n,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "contiguity",
                            contiguity,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::MatmulResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            34u32,
                            "MatmulResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyReluNd {
                        ref buf,
                        ref offset,
                        ref shape,
                        ref stride,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            35u32,
                            "ApplyReluNd",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "shape",
                            shape,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyReluNdResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            36u32,
                            "ApplyReluNdResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyRelu1dStrided {
                        ref buf,
                        ref offset,
                        ref stride,
                        ref len,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            37u32,
                            "ApplyRelu1dStrided",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyRelu1dStridedResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            38u32,
                            "ApplyRelu1dStridedResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyReluContiguous { ref buf, ref offset, ref len } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            39u32,
                            "ApplyReluContiguous",
                            0 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyReluContiguousResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            40u32,
                            "ApplyReluContiguousResponse",
                            __field0,
                        )
                    }
                    Messages::ApplySigmoidNd {
                        ref buf,
                        ref offset,
                        ref shape,
                        ref stride,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            41u32,
                            "ApplySigmoidNd",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "shape",
                            shape,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplySigmoidNdResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            42u32,
                            "ApplySigmoidNdResponse",
                            __field0,
                        )
                    }
                    Messages::ApplySigmoid1dStrided {
                        ref buf,
                        ref offset,
                        ref stride,
                        ref len,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            43u32,
                            "ApplySigmoid1dStrided",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplySigmoid1dStridedResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            44u32,
                            "ApplySigmoid1dStridedResponse",
                            __field0,
                        )
                    }
                    Messages::ApplySigmoidContiguous {
                        ref buf,
                        ref offset,
                        ref len,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            45u32,
                            "ApplySigmoidContiguous",
                            0 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplySigmoidContiguousResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            46u32,
                            "ApplySigmoidContiguousResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyTanhNd {
                        ref buf,
                        ref offset,
                        ref shape,
                        ref stride,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            47u32,
                            "ApplyTanhNd",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "shape",
                            shape,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyTanhNdResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            48u32,
                            "ApplyTanhNdResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyTanh1dStrided {
                        ref buf,
                        ref offset,
                        ref stride,
                        ref len,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            49u32,
                            "ApplyTanh1dStrided",
                            0 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "stride",
                            stride,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyTanh1dStridedResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            50u32,
                            "ApplyTanh1dStridedResponse",
                            __field0,
                        )
                    }
                    Messages::ApplyTanhContiguous { ref buf, ref offset, ref len } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            51u32,
                            "ApplyTanhContiguous",
                            0 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "buf",
                            buf,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "offset",
                            offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::ApplyTanhContiguousResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            52u32,
                            "ApplyTanhContiguousResponse",
                            __field0,
                        )
                    }
                    Messages::CopyRangeWithin {
                        ref dst,
                        ref src,
                        ref dst_offset,
                        ref src_offset,
                        ref len,
                    } => {
                        let mut __serde_state = _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "Messages",
                            53u32,
                            "CopyRangeWithin",
                            0 + 1 + 1 + 1 + 1 + 1,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "dst",
                            dst,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "src",
                            src,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "dst_offset",
                            dst_offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "src_offset",
                            src_offset,
                        )?;
                        _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "len",
                            len,
                        )?;
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    Messages::CopyRangeWithinResponse(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            54u32,
                            "CopyRangeWithinResponse",
                            __field0,
                        )
                    }
                    Messages::ActionCompleted(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "Messages",
                            55u32,
                            "ActionCompleted",
                            __field0,
                        )
                    }
                }
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for Messages {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> _serde::__private228::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __field2,
                    __field3,
                    __field4,
                    __field5,
                    __field6,
                    __field7,
                    __field8,
                    __field9,
                    __field10,
                    __field11,
                    __field12,
                    __field13,
                    __field14,
                    __field15,
                    __field16,
                    __field17,
                    __field18,
                    __field19,
                    __field20,
                    __field21,
                    __field22,
                    __field23,
                    __field24,
                    __field25,
                    __field26,
                    __field27,
                    __field28,
                    __field29,
                    __field30,
                    __field31,
                    __field32,
                    __field33,
                    __field34,
                    __field35,
                    __field36,
                    __field37,
                    __field38,
                    __field39,
                    __field40,
                    __field41,
                    __field42,
                    __field43,
                    __field44,
                    __field45,
                    __field46,
                    __field47,
                    __field48,
                    __field49,
                    __field50,
                    __field51,
                    __field52,
                    __field53,
                    __field54,
                    __field55,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "variant identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private228::Ok(__Field::__field0),
                            1u64 => _serde::__private228::Ok(__Field::__field1),
                            2u64 => _serde::__private228::Ok(__Field::__field2),
                            3u64 => _serde::__private228::Ok(__Field::__field3),
                            4u64 => _serde::__private228::Ok(__Field::__field4),
                            5u64 => _serde::__private228::Ok(__Field::__field5),
                            6u64 => _serde::__private228::Ok(__Field::__field6),
                            7u64 => _serde::__private228::Ok(__Field::__field7),
                            8u64 => _serde::__private228::Ok(__Field::__field8),
                            9u64 => _serde::__private228::Ok(__Field::__field9),
                            10u64 => _serde::__private228::Ok(__Field::__field10),
                            11u64 => _serde::__private228::Ok(__Field::__field11),
                            12u64 => _serde::__private228::Ok(__Field::__field12),
                            13u64 => _serde::__private228::Ok(__Field::__field13),
                            14u64 => _serde::__private228::Ok(__Field::__field14),
                            15u64 => _serde::__private228::Ok(__Field::__field15),
                            16u64 => _serde::__private228::Ok(__Field::__field16),
                            17u64 => _serde::__private228::Ok(__Field::__field17),
                            18u64 => _serde::__private228::Ok(__Field::__field18),
                            19u64 => _serde::__private228::Ok(__Field::__field19),
                            20u64 => _serde::__private228::Ok(__Field::__field20),
                            21u64 => _serde::__private228::Ok(__Field::__field21),
                            22u64 => _serde::__private228::Ok(__Field::__field22),
                            23u64 => _serde::__private228::Ok(__Field::__field23),
                            24u64 => _serde::__private228::Ok(__Field::__field24),
                            25u64 => _serde::__private228::Ok(__Field::__field25),
                            26u64 => _serde::__private228::Ok(__Field::__field26),
                            27u64 => _serde::__private228::Ok(__Field::__field27),
                            28u64 => _serde::__private228::Ok(__Field::__field28),
                            29u64 => _serde::__private228::Ok(__Field::__field29),
                            30u64 => _serde::__private228::Ok(__Field::__field30),
                            31u64 => _serde::__private228::Ok(__Field::__field31),
                            32u64 => _serde::__private228::Ok(__Field::__field32),
                            33u64 => _serde::__private228::Ok(__Field::__field33),
                            34u64 => _serde::__private228::Ok(__Field::__field34),
                            35u64 => _serde::__private228::Ok(__Field::__field35),
                            36u64 => _serde::__private228::Ok(__Field::__field36),
                            37u64 => _serde::__private228::Ok(__Field::__field37),
                            38u64 => _serde::__private228::Ok(__Field::__field38),
                            39u64 => _serde::__private228::Ok(__Field::__field39),
                            40u64 => _serde::__private228::Ok(__Field::__field40),
                            41u64 => _serde::__private228::Ok(__Field::__field41),
                            42u64 => _serde::__private228::Ok(__Field::__field42),
                            43u64 => _serde::__private228::Ok(__Field::__field43),
                            44u64 => _serde::__private228::Ok(__Field::__field44),
                            45u64 => _serde::__private228::Ok(__Field::__field45),
                            46u64 => _serde::__private228::Ok(__Field::__field46),
                            47u64 => _serde::__private228::Ok(__Field::__field47),
                            48u64 => _serde::__private228::Ok(__Field::__field48),
                            49u64 => _serde::__private228::Ok(__Field::__field49),
                            50u64 => _serde::__private228::Ok(__Field::__field50),
                            51u64 => _serde::__private228::Ok(__Field::__field51),
                            52u64 => _serde::__private228::Ok(__Field::__field52),
                            53u64 => _serde::__private228::Ok(__Field::__field53),
                            54u64 => _serde::__private228::Ok(__Field::__field54),
                            55u64 => _serde::__private228::Ok(__Field::__field55),
                            _ => {
                                _serde::__private228::Err(
                                    _serde::de::Error::invalid_value(
                                        _serde::de::Unexpected::Unsigned(__value),
                                        &"variant index 0 <= i < 56",
                                    ),
                                )
                            }
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "ErrorResponse" => {
                                _serde::__private228::Ok(__Field::__field0)
                            }
                            "DeviceType" => _serde::__private228::Ok(__Field::__field1),
                            "DeviceTypeResponse" => {
                                _serde::__private228::Ok(__Field::__field2)
                            }
                            "AllocFromSlice" => {
                                _serde::__private228::Ok(__Field::__field3)
                            }
                            "AllocFromSliceResponse" => {
                                _serde::__private228::Ok(__Field::__field4)
                            }
                            "Alloc" => _serde::__private228::Ok(__Field::__field5),
                            "AllocResponse" => {
                                _serde::__private228::Ok(__Field::__field6)
                            }
                            "CopyFromSlice" => {
                                _serde::__private228::Ok(__Field::__field7)
                            }
                            "CopyFromSliceResponse" => {
                                _serde::__private228::Ok(__Field::__field8)
                            }
                            "Read" => _serde::__private228::Ok(__Field::__field9),
                            "ReadResponse" => {
                                _serde::__private228::Ok(__Field::__field10)
                            }
                            "Write" => _serde::__private228::Ok(__Field::__field11),
                            "WriteResponse" => {
                                _serde::__private228::Ok(__Field::__field12)
                            }
                            "Len" => _serde::__private228::Ok(__Field::__field13),
                            "LenResponse" => _serde::__private228::Ok(__Field::__field14),
                            "Copy" => _serde::__private228::Ok(__Field::__field15),
                            "CopyResponse" => {
                                _serde::__private228::Ok(__Field::__field16)
                            }
                            "Dump" => _serde::__private228::Ok(__Field::__field17),
                            "DumpResponse" => {
                                _serde::__private228::Ok(__Field::__field18)
                            }
                            "ApplyElementwiseBinary1dStrided" => {
                                _serde::__private228::Ok(__Field::__field19)
                            }
                            "ApplyElementwiseBinary1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field20)
                            }
                            "ApplyElementwiseBinaryContiguous" => {
                                _serde::__private228::Ok(__Field::__field21)
                            }
                            "ApplyElementwiseBinaryContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field22)
                            }
                            "ApplyElementwiseBinaryNd" => {
                                _serde::__private228::Ok(__Field::__field23)
                            }
                            "ApplyElementwiseBinaryNdResponse" => {
                                _serde::__private228::Ok(__Field::__field24)
                            }
                            "Broadcast" => _serde::__private228::Ok(__Field::__field25),
                            "BroadcastResponse" => {
                                _serde::__private228::Ok(__Field::__field26)
                            }
                            "ApplyNegContiguous" => {
                                _serde::__private228::Ok(__Field::__field27)
                            }
                            "ApplyNegContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field28)
                            }
                            "ApplyNeg1dStrided" => {
                                _serde::__private228::Ok(__Field::__field29)
                            }
                            "ApplyNeg1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field30)
                            }
                            "ApplyNegNd" => _serde::__private228::Ok(__Field::__field31),
                            "ApplyNegNdResponse" => {
                                _serde::__private228::Ok(__Field::__field32)
                            }
                            "Matmul" => _serde::__private228::Ok(__Field::__field33),
                            "MatmulResponse" => {
                                _serde::__private228::Ok(__Field::__field34)
                            }
                            "ApplyReluNd" => _serde::__private228::Ok(__Field::__field35),
                            "ApplyReluNdResponse" => {
                                _serde::__private228::Ok(__Field::__field36)
                            }
                            "ApplyRelu1dStrided" => {
                                _serde::__private228::Ok(__Field::__field37)
                            }
                            "ApplyRelu1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field38)
                            }
                            "ApplyReluContiguous" => {
                                _serde::__private228::Ok(__Field::__field39)
                            }
                            "ApplyReluContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field40)
                            }
                            "ApplySigmoidNd" => {
                                _serde::__private228::Ok(__Field::__field41)
                            }
                            "ApplySigmoidNdResponse" => {
                                _serde::__private228::Ok(__Field::__field42)
                            }
                            "ApplySigmoid1dStrided" => {
                                _serde::__private228::Ok(__Field::__field43)
                            }
                            "ApplySigmoid1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field44)
                            }
                            "ApplySigmoidContiguous" => {
                                _serde::__private228::Ok(__Field::__field45)
                            }
                            "ApplySigmoidContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field46)
                            }
                            "ApplyTanhNd" => _serde::__private228::Ok(__Field::__field47),
                            "ApplyTanhNdResponse" => {
                                _serde::__private228::Ok(__Field::__field48)
                            }
                            "ApplyTanh1dStrided" => {
                                _serde::__private228::Ok(__Field::__field49)
                            }
                            "ApplyTanh1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field50)
                            }
                            "ApplyTanhContiguous" => {
                                _serde::__private228::Ok(__Field::__field51)
                            }
                            "ApplyTanhContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field52)
                            }
                            "CopyRangeWithin" => {
                                _serde::__private228::Ok(__Field::__field53)
                            }
                            "CopyRangeWithinResponse" => {
                                _serde::__private228::Ok(__Field::__field54)
                            }
                            "ActionCompleted" => {
                                _serde::__private228::Ok(__Field::__field55)
                            }
                            _ => {
                                _serde::__private228::Err(
                                    _serde::de::Error::unknown_variant(__value, VARIANTS),
                                )
                            }
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private228::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"ErrorResponse" => {
                                _serde::__private228::Ok(__Field::__field0)
                            }
                            b"DeviceType" => _serde::__private228::Ok(__Field::__field1),
                            b"DeviceTypeResponse" => {
                                _serde::__private228::Ok(__Field::__field2)
                            }
                            b"AllocFromSlice" => {
                                _serde::__private228::Ok(__Field::__field3)
                            }
                            b"AllocFromSliceResponse" => {
                                _serde::__private228::Ok(__Field::__field4)
                            }
                            b"Alloc" => _serde::__private228::Ok(__Field::__field5),
                            b"AllocResponse" => {
                                _serde::__private228::Ok(__Field::__field6)
                            }
                            b"CopyFromSlice" => {
                                _serde::__private228::Ok(__Field::__field7)
                            }
                            b"CopyFromSliceResponse" => {
                                _serde::__private228::Ok(__Field::__field8)
                            }
                            b"Read" => _serde::__private228::Ok(__Field::__field9),
                            b"ReadResponse" => {
                                _serde::__private228::Ok(__Field::__field10)
                            }
                            b"Write" => _serde::__private228::Ok(__Field::__field11),
                            b"WriteResponse" => {
                                _serde::__private228::Ok(__Field::__field12)
                            }
                            b"Len" => _serde::__private228::Ok(__Field::__field13),
                            b"LenResponse" => {
                                _serde::__private228::Ok(__Field::__field14)
                            }
                            b"Copy" => _serde::__private228::Ok(__Field::__field15),
                            b"CopyResponse" => {
                                _serde::__private228::Ok(__Field::__field16)
                            }
                            b"Dump" => _serde::__private228::Ok(__Field::__field17),
                            b"DumpResponse" => {
                                _serde::__private228::Ok(__Field::__field18)
                            }
                            b"ApplyElementwiseBinary1dStrided" => {
                                _serde::__private228::Ok(__Field::__field19)
                            }
                            b"ApplyElementwiseBinary1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field20)
                            }
                            b"ApplyElementwiseBinaryContiguous" => {
                                _serde::__private228::Ok(__Field::__field21)
                            }
                            b"ApplyElementwiseBinaryContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field22)
                            }
                            b"ApplyElementwiseBinaryNd" => {
                                _serde::__private228::Ok(__Field::__field23)
                            }
                            b"ApplyElementwiseBinaryNdResponse" => {
                                _serde::__private228::Ok(__Field::__field24)
                            }
                            b"Broadcast" => _serde::__private228::Ok(__Field::__field25),
                            b"BroadcastResponse" => {
                                _serde::__private228::Ok(__Field::__field26)
                            }
                            b"ApplyNegContiguous" => {
                                _serde::__private228::Ok(__Field::__field27)
                            }
                            b"ApplyNegContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field28)
                            }
                            b"ApplyNeg1dStrided" => {
                                _serde::__private228::Ok(__Field::__field29)
                            }
                            b"ApplyNeg1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field30)
                            }
                            b"ApplyNegNd" => _serde::__private228::Ok(__Field::__field31),
                            b"ApplyNegNdResponse" => {
                                _serde::__private228::Ok(__Field::__field32)
                            }
                            b"Matmul" => _serde::__private228::Ok(__Field::__field33),
                            b"MatmulResponse" => {
                                _serde::__private228::Ok(__Field::__field34)
                            }
                            b"ApplyReluNd" => {
                                _serde::__private228::Ok(__Field::__field35)
                            }
                            b"ApplyReluNdResponse" => {
                                _serde::__private228::Ok(__Field::__field36)
                            }
                            b"ApplyRelu1dStrided" => {
                                _serde::__private228::Ok(__Field::__field37)
                            }
                            b"ApplyRelu1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field38)
                            }
                            b"ApplyReluContiguous" => {
                                _serde::__private228::Ok(__Field::__field39)
                            }
                            b"ApplyReluContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field40)
                            }
                            b"ApplySigmoidNd" => {
                                _serde::__private228::Ok(__Field::__field41)
                            }
                            b"ApplySigmoidNdResponse" => {
                                _serde::__private228::Ok(__Field::__field42)
                            }
                            b"ApplySigmoid1dStrided" => {
                                _serde::__private228::Ok(__Field::__field43)
                            }
                            b"ApplySigmoid1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field44)
                            }
                            b"ApplySigmoidContiguous" => {
                                _serde::__private228::Ok(__Field::__field45)
                            }
                            b"ApplySigmoidContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field46)
                            }
                            b"ApplyTanhNd" => {
                                _serde::__private228::Ok(__Field::__field47)
                            }
                            b"ApplyTanhNdResponse" => {
                                _serde::__private228::Ok(__Field::__field48)
                            }
                            b"ApplyTanh1dStrided" => {
                                _serde::__private228::Ok(__Field::__field49)
                            }
                            b"ApplyTanh1dStridedResponse" => {
                                _serde::__private228::Ok(__Field::__field50)
                            }
                            b"ApplyTanhContiguous" => {
                                _serde::__private228::Ok(__Field::__field51)
                            }
                            b"ApplyTanhContiguousResponse" => {
                                _serde::__private228::Ok(__Field::__field52)
                            }
                            b"CopyRangeWithin" => {
                                _serde::__private228::Ok(__Field::__field53)
                            }
                            b"CopyRangeWithinResponse" => {
                                _serde::__private228::Ok(__Field::__field54)
                            }
                            b"ActionCompleted" => {
                                _serde::__private228::Ok(__Field::__field55)
                            }
                            _ => {
                                let __value = &_serde::__private228::from_utf8_lossy(
                                    __value,
                                );
                                _serde::__private228::Err(
                                    _serde::de::Error::unknown_variant(__value, VARIANTS),
                                )
                            }
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private228::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de> {
                    marker: _serde::__private228::PhantomData<Messages>,
                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = Messages;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private228::Formatter,
                    ) -> _serde::__private228::fmt::Result {
                        _serde::__private228::Formatter::write_str(
                            __formatter,
                            "enum Messages",
                        )
                    }
                    fn visit_enum<__A>(
                        self,
                        __data: __A,
                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::EnumAccess<'de>,
                    {
                        match _serde::de::EnumAccess::variant(__data)? {
                            (__Field::__field0, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "message" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"message" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ErrorResponse",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            String,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ErrorResponse with 1 element",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ErrorResponse {
                                            message: __field0,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<String> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "message",
                                                            ),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<String>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("message")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ErrorResponse {
                                            message: __field0,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["message"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field1, __variant) => {
                                _serde::de::VariantAccess::unit_variant(__variant)?;
                                _serde::__private228::Ok(Messages::DeviceType)
                            }
                            (__Field::__field2, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "device_type" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"device_type" => {
                                                _serde::__private228::Ok(__Field::__field0)
                                            }
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::DeviceTypeResponse",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            DeviceType,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::DeviceTypeResponse with 1 element",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::DeviceTypeResponse {
                                            device_type: __field0,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            DeviceType,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "device_type",
                                                            ),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<DeviceType>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("device_type")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::DeviceTypeResponse {
                                            device_type: __field0,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["device_type"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field3, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "src" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"src" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::AllocFromSlice",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            Slice,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::AllocFromSlice with 1 element",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::AllocFromSlice {
                                            src: __field0,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<Slice> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("src"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Slice>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("src")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::AllocFromSlice {
                                            src: __field0,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["src"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field4, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<TypelessBuf, TensorError>,
                                    >(__variant),
                                    Messages::AllocFromSliceResponse,
                                )
                            }
                            (__Field::__field5, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "len" => _serde::__private228::Ok(__Field::__field0),
                                            "dtype" => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"len" => _serde::__private228::Ok(__Field::__field0),
                                            b"dtype" => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::Alloc",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::Alloc with 2 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            DType,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::Alloc with 2 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Alloc {
                                            len: __field0,
                                            dtype: __field1,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<DType> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("dtype"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<DType>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("dtype")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Alloc {
                                            len: __field0,
                                            dtype: __field1,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["len", "dtype"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field6, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<TypelessBuf, TensorError>,
                                    >(__variant),
                                    Messages::AllocResponse,
                                )
                            }
                            (__Field::__field7, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "dst" => _serde::__private228::Ok(__Field::__field0),
                                            "src" => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"dst" => _serde::__private228::Ok(__Field::__field0),
                                            b"src" => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::CopyFromSlice",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::CopyFromSlice with 2 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            Slice,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::CopyFromSlice with 2 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::CopyFromSlice {
                                            dst: __field0,
                                            src: __field1,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<Slice> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("dst"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("src"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Slice>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("dst")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("src")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::CopyFromSlice {
                                            dst: __field0,
                                            src: __field1,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["dst", "src"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field8, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::CopyFromSliceResponse,
                                )
                            }
                            (__Field::__field9, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::Read",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::Read with 2 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::Read with 2 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Read {
                                            buf: __field0,
                                            offset: __field1,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Read {
                                            buf: __field0,
                                            offset: __field1,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["buf", "offset"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field10, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<Value, TensorError>,
                                    >(__variant),
                                    Messages::ReadResponse,
                                )
                            }
                            (__Field::__field11, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "value" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"value" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::Write",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::Write with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::Write with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            Value,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::Write with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Write {
                                            buf: __field0,
                                            offset: __field1,
                                            value: __field2,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<Value> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("value"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Value>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("value")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Write {
                                            buf: __field0,
                                            offset: __field1,
                                            value: __field2,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "value",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field12, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::WriteResponse,
                                )
                            }
                            (__Field::__field13, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::Len",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::Len with 1 element",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Len { buf: __field0 })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Len { buf: __field0 })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["buf"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field14, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        usize,
                                    >(__variant),
                                    Messages::LenResponse,
                                )
                            }
                            (__Field::__field15, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "src" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"src" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::Copy",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::Copy with 1 element",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Copy { src: __field0 })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("src"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("src")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Copy { src: __field0 })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["src"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field16, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<TypelessBuf, TensorError>,
                                    >(__variant),
                                    Messages::CopyResponse,
                                )
                            }
                            (__Field::__field17, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "src" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"src" => _serde::__private228::Ok(__Field::__field0),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::Dump",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::Dump with 1 element",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Dump { src: __field0 })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("src"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("src")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Dump { src: __field0 })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &["src"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field18, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<Slice, TensorError>,
                                    >(__variant),
                                    Messages::DumpResponse,
                                )
                            }
                            (__Field::__field19, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __field4,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            4u64 => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "op" => _serde::__private228::Ok(__Field::__field1),
                                            "offset" => _serde::__private228::Ok(__Field::__field2),
                                            "stride" => _serde::__private228::Ok(__Field::__field3),
                                            "len" => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"op" => _serde::__private228::Ok(__Field::__field1),
                                            b"offset" => _serde::__private228::Ok(__Field::__field2),
                                            b"stride" => _serde::__private228::Ok(__Field::__field3),
                                            b"len" => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyElementwiseBinary1dStrided",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyElementwiseBinary1dStrided with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            (BinaryOpType, Value),
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyElementwiseBinary1dStrided with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyElementwiseBinary1dStrided with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            isize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyElementwiseBinary1dStrided with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field4 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        4usize,
                                                        &"struct variant Messages::ApplyElementwiseBinary1dStrided with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyElementwiseBinary1dStrided {
                                            buf: __field0,
                                            op: __field1,
                                            offset: __field2,
                                            stride: __field3,
                                            len: __field4,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<
                                            (BinaryOpType, Value),
                                        > = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<isize> = _serde::__private228::None;
                                        let mut __field4: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("op"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            (BinaryOpType, Value),
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<isize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field4 => {
                                                    if _serde::__private228::Option::is_some(&__field4) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field4 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("op")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        let __field4 = match __field4 {
                                            _serde::__private228::Some(__field4) => __field4,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyElementwiseBinary1dStrided {
                                            buf: __field0,
                                            op: __field1,
                                            offset: __field2,
                                            stride: __field3,
                                            len: __field4,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "op",
                                    "offset",
                                    "stride",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field20, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyElementwiseBinary1dStridedResponse,
                                )
                            }
                            (__Field::__field21, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "op" => _serde::__private228::Ok(__Field::__field1),
                                            "start" => _serde::__private228::Ok(__Field::__field2),
                                            "len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"op" => _serde::__private228::Ok(__Field::__field1),
                                            b"start" => _serde::__private228::Ok(__Field::__field2),
                                            b"len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyElementwiseBinaryContiguous",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryContiguous with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            (BinaryOpType, Value),
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryContiguous with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryContiguous with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryContiguous with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyElementwiseBinaryContiguous {
                                            buf: __field0,
                                            op: __field1,
                                            start: __field2,
                                            len: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<
                                            (BinaryOpType, Value),
                                        > = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("op"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            (BinaryOpType, Value),
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("start"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("op")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("start")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyElementwiseBinaryContiguous {
                                            buf: __field0,
                                            op: __field1,
                                            start: __field2,
                                            len: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "op",
                                    "start",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field22, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyElementwiseBinaryContiguousResponse,
                                )
                            }
                            (__Field::__field23, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __field4,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            4u64 => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "op" => _serde::__private228::Ok(__Field::__field1),
                                            "offset" => _serde::__private228::Ok(__Field::__field2),
                                            "shape" => _serde::__private228::Ok(__Field::__field3),
                                            "stride" => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"op" => _serde::__private228::Ok(__Field::__field1),
                                            b"offset" => _serde::__private228::Ok(__Field::__field2),
                                            b"shape" => _serde::__private228::Ok(__Field::__field3),
                                            b"stride" => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyElementwiseBinaryNd",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryNd with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            (BinaryOpType, Value),
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryNd with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryNd with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            Vec<usize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryNd with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field4 = match _serde::de::SeqAccess::next_element::<
                                            Vec<isize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        4usize,
                                                        &"struct variant Messages::ApplyElementwiseBinaryNd with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyElementwiseBinaryNd {
                                            buf: __field0,
                                            op: __field1,
                                            offset: __field2,
                                            shape: __field3,
                                            stride: __field4,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<
                                            (BinaryOpType, Value),
                                        > = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<
                                            Vec<usize>,
                                        > = _serde::__private228::None;
                                        let mut __field4: _serde::__private228::Option<
                                            Vec<isize>,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("op"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            (BinaryOpType, Value),
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("shape"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<usize>>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field4 => {
                                                    if _serde::__private228::Option::is_some(&__field4) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field4 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<isize>>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("op")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("shape")?
                                            }
                                        };
                                        let __field4 = match __field4 {
                                            _serde::__private228::Some(__field4) => __field4,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyElementwiseBinaryNd {
                                            buf: __field0,
                                            op: __field1,
                                            offset: __field2,
                                            shape: __field3,
                                            stride: __field4,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "op",
                                    "offset",
                                    "shape",
                                    "stride",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field24, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyElementwiseBinaryNdResponse,
                                )
                            }
                            (__Field::__field25, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "left" => _serde::__private228::Ok(__Field::__field0),
                                            "right" => _serde::__private228::Ok(__Field::__field1),
                                            "dst" => _serde::__private228::Ok(__Field::__field2),
                                            "op" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"left" => _serde::__private228::Ok(__Field::__field0),
                                            b"right" => _serde::__private228::Ok(__Field::__field1),
                                            b"dst" => _serde::__private228::Ok(__Field::__field2),
                                            b"op" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::Broadcast",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            (TypelessBuf, MetaTensor),
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::Broadcast with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            (TypelessBuf, MetaTensor),
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::Broadcast with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            (TypelessBuf, MetaTensor),
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::Broadcast with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            BinaryOpType,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::Broadcast with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Broadcast {
                                            left: __field0,
                                            right: __field1,
                                            dst: __field2,
                                            op: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            (TypelessBuf, MetaTensor),
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<
                                            (TypelessBuf, MetaTensor),
                                        > = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<
                                            (TypelessBuf, MetaTensor),
                                        > = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<
                                            BinaryOpType,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("left"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            (TypelessBuf, MetaTensor),
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("right"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            (TypelessBuf, MetaTensor),
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("dst"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            (TypelessBuf, MetaTensor),
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("op"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            BinaryOpType,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("left")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("right")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("dst")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("op")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Broadcast {
                                            left: __field0,
                                            right: __field1,
                                            dst: __field2,
                                            op: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "left",
                                    "right",
                                    "dst",
                                    "op",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field26, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::BroadcastResponse,
                                )
                            }
                            (__Field::__field27, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "start" => _serde::__private228::Ok(__Field::__field1),
                                            "len" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"start" => _serde::__private228::Ok(__Field::__field1),
                                            b"len" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyNegContiguous",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyNegContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyNegContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyNegContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyNegContiguous {
                                            buf: __field0,
                                            start: __field1,
                                            len: __field2,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("start"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("start")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyNegContiguous {
                                            buf: __field0,
                                            start: __field1,
                                            len: __field2,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "start",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field28, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyNegContiguousResponse,
                                )
                            }
                            (__Field::__field29, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "stride" => _serde::__private228::Ok(__Field::__field2),
                                            "len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"stride" => _serde::__private228::Ok(__Field::__field2),
                                            b"len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyNeg1dStrided",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyNeg1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyNeg1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            isize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyNeg1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyNeg1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyNeg1dStrided {
                                            buf: __field0,
                                            offset: __field1,
                                            stride: __field2,
                                            len: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<isize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<isize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyNeg1dStrided {
                                            buf: __field0,
                                            offset: __field1,
                                            stride: __field2,
                                            len: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "stride",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field30, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyNeg1dStridedResponse,
                                )
                            }
                            (__Field::__field31, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "shape" => _serde::__private228::Ok(__Field::__field2),
                                            "stride" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"shape" => _serde::__private228::Ok(__Field::__field2),
                                            b"stride" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyNegNd",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyNegNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyNegNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            Vec<usize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyNegNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            Vec<isize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyNegNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyNegNd {
                                            buf: __field0,
                                            offset: __field1,
                                            shape: __field2,
                                            stride: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<
                                            Vec<usize>,
                                        > = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<
                                            Vec<isize>,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("shape"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<usize>>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<isize>>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("shape")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyNegNd {
                                            buf: __field0,
                                            offset: __field1,
                                            shape: __field2,
                                            stride: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "shape",
                                    "stride",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field32, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyNegNdResponse,
                                )
                            }
                            (__Field::__field33, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __field4,
                                    __field5,
                                    __field6,
                                    __field7,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            4u64 => _serde::__private228::Ok(__Field::__field4),
                                            5u64 => _serde::__private228::Ok(__Field::__field5),
                                            6u64 => _serde::__private228::Ok(__Field::__field6),
                                            7u64 => _serde::__private228::Ok(__Field::__field7),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "lhs" => _serde::__private228::Ok(__Field::__field0),
                                            "rhs" => _serde::__private228::Ok(__Field::__field1),
                                            "dst" => _serde::__private228::Ok(__Field::__field2),
                                            "b" => _serde::__private228::Ok(__Field::__field3),
                                            "m" => _serde::__private228::Ok(__Field::__field4),
                                            "k" => _serde::__private228::Ok(__Field::__field5),
                                            "n" => _serde::__private228::Ok(__Field::__field6),
                                            "contiguity" => _serde::__private228::Ok(__Field::__field7),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"lhs" => _serde::__private228::Ok(__Field::__field0),
                                            b"rhs" => _serde::__private228::Ok(__Field::__field1),
                                            b"dst" => _serde::__private228::Ok(__Field::__field2),
                                            b"b" => _serde::__private228::Ok(__Field::__field3),
                                            b"m" => _serde::__private228::Ok(__Field::__field4),
                                            b"k" => _serde::__private228::Ok(__Field::__field5),
                                            b"n" => _serde::__private228::Ok(__Field::__field6),
                                            b"contiguity" => _serde::__private228::Ok(__Field::__field7),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::Matmul",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            (TypelessBuf, MetaTensor),
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::Matmul with 8 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            (TypelessBuf, MetaTensor),
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::Matmul with 8 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::Matmul with 8 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::Matmul with 8 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field4 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        4usize,
                                                        &"struct variant Messages::Matmul with 8 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field5 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        5usize,
                                                        &"struct variant Messages::Matmul with 8 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field6 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        6usize,
                                                        &"struct variant Messages::Matmul with 8 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field7 = match _serde::de::SeqAccess::next_element::<
                                            ContiguityTypes,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        7usize,
                                                        &"struct variant Messages::Matmul with 8 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Matmul {
                                            lhs: __field0,
                                            rhs: __field1,
                                            dst: __field2,
                                            b: __field3,
                                            m: __field4,
                                            k: __field5,
                                            n: __field6,
                                            contiguity: __field7,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            (TypelessBuf, MetaTensor),
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<
                                            (TypelessBuf, MetaTensor),
                                        > = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field4: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field5: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field6: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field7: _serde::__private228::Option<
                                            ContiguityTypes,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("lhs"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            (TypelessBuf, MetaTensor),
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("rhs"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            (TypelessBuf, MetaTensor),
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("dst"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("b"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field4 => {
                                                    if _serde::__private228::Option::is_some(&__field4) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("m"),
                                                        );
                                                    }
                                                    __field4 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field5 => {
                                                    if _serde::__private228::Option::is_some(&__field5) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("k"),
                                                        );
                                                    }
                                                    __field5 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field6 => {
                                                    if _serde::__private228::Option::is_some(&__field6) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("n"),
                                                        );
                                                    }
                                                    __field6 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field7 => {
                                                    if _serde::__private228::Option::is_some(&__field7) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "contiguity",
                                                            ),
                                                        );
                                                    }
                                                    __field7 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            ContiguityTypes,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("lhs")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("rhs")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("dst")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("b")?
                                            }
                                        };
                                        let __field4 = match __field4 {
                                            _serde::__private228::Some(__field4) => __field4,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("m")?
                                            }
                                        };
                                        let __field5 = match __field5 {
                                            _serde::__private228::Some(__field5) => __field5,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("k")?
                                            }
                                        };
                                        let __field6 = match __field6 {
                                            _serde::__private228::Some(__field6) => __field6,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("n")?
                                            }
                                        };
                                        let __field7 = match __field7 {
                                            _serde::__private228::Some(__field7) => __field7,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("contiguity")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::Matmul {
                                            lhs: __field0,
                                            rhs: __field1,
                                            dst: __field2,
                                            b: __field3,
                                            m: __field4,
                                            k: __field5,
                                            n: __field6,
                                            contiguity: __field7,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "lhs",
                                    "rhs",
                                    "dst",
                                    "b",
                                    "m",
                                    "k",
                                    "n",
                                    "contiguity",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field34, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::MatmulResponse,
                                )
                            }
                            (__Field::__field35, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "shape" => _serde::__private228::Ok(__Field::__field2),
                                            "stride" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"shape" => _serde::__private228::Ok(__Field::__field2),
                                            b"stride" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyReluNd",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyReluNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyReluNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            Vec<usize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyReluNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            Vec<isize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyReluNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyReluNd {
                                            buf: __field0,
                                            offset: __field1,
                                            shape: __field2,
                                            stride: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<
                                            Vec<usize>,
                                        > = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<
                                            Vec<isize>,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("shape"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<usize>>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<isize>>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("shape")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyReluNd {
                                            buf: __field0,
                                            offset: __field1,
                                            shape: __field2,
                                            stride: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "shape",
                                    "stride",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field36, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyReluNdResponse,
                                )
                            }
                            (__Field::__field37, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "stride" => _serde::__private228::Ok(__Field::__field2),
                                            "len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"stride" => _serde::__private228::Ok(__Field::__field2),
                                            b"len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyRelu1dStrided",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyRelu1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyRelu1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            isize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyRelu1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyRelu1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyRelu1dStrided {
                                            buf: __field0,
                                            offset: __field1,
                                            stride: __field2,
                                            len: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<isize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<isize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyRelu1dStrided {
                                            buf: __field0,
                                            offset: __field1,
                                            stride: __field2,
                                            len: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "stride",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field38, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyRelu1dStridedResponse,
                                )
                            }
                            (__Field::__field39, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "len" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"len" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyReluContiguous",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyReluContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyReluContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyReluContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyReluContiguous {
                                            buf: __field0,
                                            offset: __field1,
                                            len: __field2,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyReluContiguous {
                                            buf: __field0,
                                            offset: __field1,
                                            len: __field2,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field40, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyReluContiguousResponse,
                                )
                            }
                            (__Field::__field41, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "shape" => _serde::__private228::Ok(__Field::__field2),
                                            "stride" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"shape" => _serde::__private228::Ok(__Field::__field2),
                                            b"stride" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplySigmoidNd",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplySigmoidNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplySigmoidNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            Vec<usize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplySigmoidNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            Vec<isize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplySigmoidNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplySigmoidNd {
                                            buf: __field0,
                                            offset: __field1,
                                            shape: __field2,
                                            stride: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<
                                            Vec<usize>,
                                        > = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<
                                            Vec<isize>,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("shape"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<usize>>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<isize>>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("shape")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplySigmoidNd {
                                            buf: __field0,
                                            offset: __field1,
                                            shape: __field2,
                                            stride: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "shape",
                                    "stride",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field42, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplySigmoidNdResponse,
                                )
                            }
                            (__Field::__field43, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "stride" => _serde::__private228::Ok(__Field::__field2),
                                            "len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"stride" => _serde::__private228::Ok(__Field::__field2),
                                            b"len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplySigmoid1dStrided",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplySigmoid1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplySigmoid1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            isize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplySigmoid1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplySigmoid1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplySigmoid1dStrided {
                                            buf: __field0,
                                            offset: __field1,
                                            stride: __field2,
                                            len: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<isize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<isize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplySigmoid1dStrided {
                                            buf: __field0,
                                            offset: __field1,
                                            stride: __field2,
                                            len: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "stride",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field44, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplySigmoid1dStridedResponse,
                                )
                            }
                            (__Field::__field45, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "len" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"len" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplySigmoidContiguous",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplySigmoidContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplySigmoidContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplySigmoidContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplySigmoidContiguous {
                                            buf: __field0,
                                            offset: __field1,
                                            len: __field2,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplySigmoidContiguous {
                                            buf: __field0,
                                            offset: __field1,
                                            len: __field2,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field46, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplySigmoidContiguousResponse,
                                )
                            }
                            (__Field::__field47, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "shape" => _serde::__private228::Ok(__Field::__field2),
                                            "stride" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"shape" => _serde::__private228::Ok(__Field::__field2),
                                            b"stride" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyTanhNd",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyTanhNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyTanhNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            Vec<usize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyTanhNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            Vec<isize>,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyTanhNd with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyTanhNd {
                                            buf: __field0,
                                            offset: __field1,
                                            shape: __field2,
                                            stride: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<
                                            Vec<usize>,
                                        > = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<
                                            Vec<isize>,
                                        > = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("shape"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<usize>>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<Vec<isize>>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("shape")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyTanhNd {
                                            buf: __field0,
                                            offset: __field1,
                                            shape: __field2,
                                            stride: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "shape",
                                    "stride",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field48, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyTanhNdResponse,
                                )
                            }
                            (__Field::__field49, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "stride" => _serde::__private228::Ok(__Field::__field2),
                                            "len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"stride" => _serde::__private228::Ok(__Field::__field2),
                                            b"len" => _serde::__private228::Ok(__Field::__field3),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyTanh1dStrided",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyTanh1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyTanh1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            isize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyTanh1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::ApplyTanh1dStrided with 4 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyTanh1dStrided {
                                            buf: __field0,
                                            offset: __field1,
                                            stride: __field2,
                                            len: __field3,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<isize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<isize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("stride")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyTanh1dStrided {
                                            buf: __field0,
                                            offset: __field1,
                                            stride: __field2,
                                            len: __field3,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "stride",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field50, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyTanh1dStridedResponse,
                                )
                            }
                            (__Field::__field51, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "buf" => _serde::__private228::Ok(__Field::__field0),
                                            "offset" => _serde::__private228::Ok(__Field::__field1),
                                            "len" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"buf" => _serde::__private228::Ok(__Field::__field0),
                                            b"offset" => _serde::__private228::Ok(__Field::__field1),
                                            b"len" => _serde::__private228::Ok(__Field::__field2),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::ApplyTanhContiguous",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::ApplyTanhContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::ApplyTanhContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::ApplyTanhContiguous with 3 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyTanhContiguous {
                                            buf: __field0,
                                            offset: __field1,
                                            len: __field2,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("buf"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("offset"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("buf")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("offset")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::ApplyTanhContiguous {
                                            buf: __field0,
                                            offset: __field1,
                                            len: __field2,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "buf",
                                    "offset",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field52, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::ApplyTanhContiguousResponse,
                                )
                            }
                            (__Field::__field53, __variant) => {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __field4,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            4u64 => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "dst" => _serde::__private228::Ok(__Field::__field0),
                                            "src" => _serde::__private228::Ok(__Field::__field1),
                                            "dst_offset" => _serde::__private228::Ok(__Field::__field2),
                                            "src_offset" => _serde::__private228::Ok(__Field::__field3),
                                            "len" => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"dst" => _serde::__private228::Ok(__Field::__field0),
                                            b"src" => _serde::__private228::Ok(__Field::__field1),
                                            b"dst_offset" => _serde::__private228::Ok(__Field::__field2),
                                            b"src_offset" => _serde::__private228::Ok(__Field::__field3),
                                            b"len" => _serde::__private228::Ok(__Field::__field4),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<Messages>,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = Messages;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct variant Messages::CopyRangeWithin",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct variant Messages::CopyRangeWithin with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            TypelessBuf,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct variant Messages::CopyRangeWithin with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct variant Messages::CopyRangeWithin with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct variant Messages::CopyRangeWithin with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field4 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        4usize,
                                                        &"struct variant Messages::CopyRangeWithin with 5 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::CopyRangeWithin {
                                            dst: __field0,
                                            src: __field1,
                                            dst_offset: __field2,
                                            src_offset: __field3,
                                            len: __field4,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<
                                            TypelessBuf,
                                        > = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field4: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("dst"),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("src"),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<
                                                            TypelessBuf,
                                                        >(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "dst_offset",
                                                            ),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "src_offset",
                                                            ),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field4 => {
                                                    if _serde::__private228::Option::is_some(&__field4) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field("len"),
                                                        );
                                                    }
                                                    __field4 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("dst")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("src")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("dst_offset")?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("src_offset")?
                                            }
                                        };
                                        let __field4 = match __field4 {
                                            _serde::__private228::Some(__field4) => __field4,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("len")?
                                            }
                                        };
                                        _serde::__private228::Ok(Messages::CopyRangeWithin {
                                            dst: __field0,
                                            src: __field1,
                                            dst_offset: __field2,
                                            src_offset: __field3,
                                            len: __field4,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "dst",
                                    "src",
                                    "dst_offset",
                                    "src_offset",
                                    "len",
                                ];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<Messages>,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field54, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        Result<(), TensorError>,
                                    >(__variant),
                                    Messages::CopyRangeWithinResponse,
                                )
                            }
                            (__Field::__field55, __variant) => {
                                _serde::__private228::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        u32,
                                    >(__variant),
                                    Messages::ActionCompleted,
                                )
                            }
                        }
                    }
                }
                #[doc(hidden)]
                const VARIANTS: &'static [&'static str] = &[
                    "ErrorResponse",
                    "DeviceType",
                    "DeviceTypeResponse",
                    "AllocFromSlice",
                    "AllocFromSliceResponse",
                    "Alloc",
                    "AllocResponse",
                    "CopyFromSlice",
                    "CopyFromSliceResponse",
                    "Read",
                    "ReadResponse",
                    "Write",
                    "WriteResponse",
                    "Len",
                    "LenResponse",
                    "Copy",
                    "CopyResponse",
                    "Dump",
                    "DumpResponse",
                    "ApplyElementwiseBinary1dStrided",
                    "ApplyElementwiseBinary1dStridedResponse",
                    "ApplyElementwiseBinaryContiguous",
                    "ApplyElementwiseBinaryContiguousResponse",
                    "ApplyElementwiseBinaryNd",
                    "ApplyElementwiseBinaryNdResponse",
                    "Broadcast",
                    "BroadcastResponse",
                    "ApplyNegContiguous",
                    "ApplyNegContiguousResponse",
                    "ApplyNeg1dStrided",
                    "ApplyNeg1dStridedResponse",
                    "ApplyNegNd",
                    "ApplyNegNdResponse",
                    "Matmul",
                    "MatmulResponse",
                    "ApplyReluNd",
                    "ApplyReluNdResponse",
                    "ApplyRelu1dStrided",
                    "ApplyRelu1dStridedResponse",
                    "ApplyReluContiguous",
                    "ApplyReluContiguousResponse",
                    "ApplySigmoidNd",
                    "ApplySigmoidNdResponse",
                    "ApplySigmoid1dStrided",
                    "ApplySigmoid1dStridedResponse",
                    "ApplySigmoidContiguous",
                    "ApplySigmoidContiguousResponse",
                    "ApplyTanhNd",
                    "ApplyTanhNdResponse",
                    "ApplyTanh1dStrided",
                    "ApplyTanh1dStridedResponse",
                    "ApplyTanhContiguous",
                    "ApplyTanhContiguousResponse",
                    "CopyRangeWithin",
                    "CopyRangeWithinResponse",
                    "ActionCompleted",
                ];
                _serde::Deserializer::deserialize_enum(
                    __deserializer,
                    "Messages",
                    VARIANTS,
                    __Visitor {
                        marker: _serde::__private228::PhantomData::<Messages>,
                        lifetime: _serde::__private228::PhantomData,
                    },
                )
            }
        }
    };
    impl Messages {
        #[inline(always)]
        pub fn is_response(&self) -> bool {
            match self {
                Messages::DeviceTypeResponse { .. }
                | Messages::AllocFromSliceResponse { .. }
                | Messages::AllocResponse { .. }
                | Messages::CopyFromSliceResponse { .. }
                | Messages::ReadResponse { .. }
                | Messages::WriteResponse { .. }
                | Messages::LenResponse { .. }
                | Messages::CopyResponse { .. }
                | Messages::DumpResponse { .. }
                | Messages::ApplyElementwiseBinary1dStridedResponse { .. }
                | Messages::ApplyElementwiseBinaryContiguousResponse { .. }
                | Messages::ApplyElementwiseBinaryNdResponse { .. }
                | Messages::BroadcastResponse { .. }
                | Messages::MatmulResponse { .. }
                | Messages::ApplyNeg1dStridedResponse { .. }
                | Messages::ApplyNegContiguousResponse { .. }
                | Messages::ApplyNegNdResponse { .. }
                | Messages::ErrorResponse { .. }
                | Messages::ActionCompleted { .. }
                | Messages::ApplyReluNdResponse { .. }
                | Messages::ApplyRelu1dStridedResponse { .. }
                | Messages::ApplyReluContiguousResponse { .. }
                | Messages::ApplySigmoidNdResponse { .. }
                | Messages::ApplySigmoid1dStridedResponse { .. }
                | Messages::ApplySigmoidContiguousResponse { .. }
                | Messages::ApplyTanhNdResponse { .. }
                | Messages::ApplyTanh1dStridedResponse { .. }
                | Messages::ApplyTanhContiguousResponse { .. }
                | Messages::CopyRangeWithinResponse { .. } => true,
                _ => false,
            }
        }
    }
}
